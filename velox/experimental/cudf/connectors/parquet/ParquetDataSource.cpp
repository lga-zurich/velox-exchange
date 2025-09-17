/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/connectors/parquet/ParquetConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnectorSplit.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/time/Timer.h"
#include "velox/expression/FieldReference.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <memory>
#include <regex>
#include <string>

namespace facebook::velox::cudf_velox::connector::parquet {

using namespace facebook::velox::connector;

ParquetDataSource::ParquetDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<ParquetConfig>& parquetConfig)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241}, // Parquet blue,
          std::nullopt,
          fmt::format("[{}]", tableHandle->name())),
      parquetConfig_(parquetConfig),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      pool_(connectorQueryCtx->memoryPool()),
      outputType_(outputType),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  // Set up column projection if needed
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const ParquetColumnHandle*>(it->second.get());
    readColumnNames_.emplace_back(handle->name());
  }

  // Dynamic cast tableHandle to ParquetTableHandle
  tableHandle_ =
      std::dynamic_pointer_cast<const ParquetTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of ParquetTableHandle");

  // Create empty IOStats for later use
  ioStats_ = std::make_shared<io::IoStatistics>();

  // Create subfield filter
  auto subfieldFilter = tableHandle_->subfieldFilterExpr();
  if (subfieldFilter) {
    subfieldFilterExprSet_ = expressionEvaluator_->compile(subfieldFilter);
    // Add fields in the filter to the columns to read if not there
    for (const auto& field : subfieldFilterExprSet_->distinctFields()) {
      if (std::find(
              readColumnNames_.begin(),
              readColumnNames_.end(),
              field->name()) == readColumnNames_.end()) {
        LOG(INFO) << "Adding sub field to readColumnNames_ : " << field->name();
        readColumnNames_.push_back(field->name());
      }
    }
  }

  // Create remaining filter
  auto remainingFilter = tableHandle_->remainingFilter();
  if (remainingFilter) {
    remainingFilterExprSet_ = expressionEvaluator_->compile(remainingFilter);
    for (const auto& field : remainingFilterExprSet_->distinctFields()) {
      // Add fields in the filter to the columns to read if not there
      if (std::find(
              readColumnNames_.begin(),
              readColumnNames_.end(),
              field->name()) == readColumnNames_.end()) {
        LOG(INFO) << "Adding remaining field to readColumnNames_ : "
                  << field->name();
        readColumnNames_.push_back(field->name());
      }
    }

    const RowTypePtr remainingFilterType_ = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> new_names;
        std::vector<TypePtr> new_types;

        for (const auto& name : readColumnNames_) {
          auto parsedType = tableHandle_->dataColumns()->findChild(name);
          new_names.emplace_back(std::move(name));
          new_types.push_back(parsedType);
        }

        return ROW(std::move(new_names), std::move(new_types));
      } else {
        return outputType_;
      }
    }();

    cudfExpressionEvaluator_ = velox::cudf_velox::ExpressionEvaluator(
        remainingFilterExprSet_->exprs(), remainingFilterType_);
    // TODO(kn): Get column names and subfields from remaining filter and add to
    // readColumnNames_
  }
}

std::optional<RowVectorPtr> ParquetDataSource::next(
    uint64_t /*size*/,
    velox::ContinueFuture& /* future */) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  // Basic sanity checks
  VELOX_CHECK_NOT_NULL(split_, "No split to process. Call addSplit first.");
  if (FLAGS_velox_cudf_zrl_reader)
    VELOX_CHECK_NOT_NULL(splitReaderZrl_, "No split reader present");
  else
    VELOX_CHECK_NOT_NULL(splitReaderCudf_, "No split reader present");

  std::unique_ptr<cudf::table> cudfTable;
  size_t read_bytes;

  // Record start time before reading chunk
  auto startTimeUs = getCurrentTimeMicro();

  if (FLAGS_velox_cudf_zrl_reader) {
    LOG(INFO) << "Getting next chunk";
    if (chunks_.size() == 0) {
      LOG(INFO) << "NO CHUNK";
      return nullptr;
    }
    uint32_t chunk = chunks_.back();
    chunks_.pop_back();
    LOG(INFO) << "Popping Chunk is : " << chunk;

    cudfTable = [&] {
      auto noLikeExpr = std::vector<
          ibm::velox::cudf_velox::connector::parquet_hack::LikeExpr>();
      if (subfieldFilterExprSet_) {
        return splitReaderZrl_->LoadTableSeq(
            readColumnNames_,
            readColumnNames_,
            chunk,
            subfieldTree_.back(),
            noLikeExpr,
            read_bytes,
            stream_);
      } else {
        return splitReaderZrl_->LoadTableSeq(
            readColumnNames_,
            readColumnNames_,
            chunk,
            std::nullopt,
            noLikeExpr,
            read_bytes,
            stream_);
      }
    }();

    LOG(INFO) << "Read Chunk: " << chunk;

  } else {
    if (not splitReaderCudf_->has_next()) {
      return nullptr;
    }
    // Read a table chunk
    auto [table, metadata] = splitReaderCudf_->read_chunk();
    cudfTable = std::move(table);
    // Fill in the column names if reading the first chunk.
    if (columnNames_.empty()) {
      for (auto schema : metadata.schema_info) {
        columnNames_.emplace_back(schema.name);
      }
    }
  }
  completedBytes_ += read_bytes;

  TotalScanTimeCallbackData* callbackData =
      new TotalScanTimeCallbackData{startTimeUs, ioStats_};

  // Launch host callback to calculate timing when scan completes
  cudaLaunchHostFunc(
      stream_.value(),
      &ParquetDataSource::totalScanTimeCalculator,
      callbackData);

  uint64_t filterTimeUs{0};
  // Apply remaining filter if present
  if (remainingFilterExprSet_) {
    MicrosecondTimer filterTimer(&filterTimeUs);
    auto cudfTableColumns = cudfTable->release();
    const auto originalNumColumns = cudfTableColumns.size();
    // Filter may need addtional computed columns which are added to
    // cudfTableColumns
    auto filterResult = cudfExpressionEvaluator_.compute(
        cudfTableColumns, stream_, cudf::get_current_device_resource_ref());
    // discard computed columns
    std::vector<std::unique_ptr<cudf::column>> originalColumns;
    originalColumns.reserve(originalNumColumns);
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + originalNumColumns,
        std::back_inserter(originalColumns));
    auto originalTable =
        std::make_unique<cudf::table>(std::move(originalColumns));
    // Keep only rows where the filter is true
    cudfTable = cudf::apply_boolean_mask(
        *originalTable,
        *filterResult[0],
        stream_,
        cudf::get_current_device_resource_ref());
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);

  // Output RowVectorPtr
  const auto nRows = cudfTable->num_rows();

  // keep only outputType_.size() columns in cudfTable_
  if (outputType_->size() < cudfTable->num_columns()) {
    auto cudfTableColumns = cudfTable->release();
    std::vector<std::unique_ptr<cudf::column>> originalColumns;
    originalColumns.reserve(outputType_->size());
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + outputType_->size(),
        std::back_inserter(originalColumns));
    cudfTable = std::make_unique<cudf::table>(std::move(originalColumns));
  }

  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, nRows, std::move(cudfTable), stream_)
      : with_arrow::toVeloxColumn(
            cudfTable->view(), pool_, outputType_->names(), stream_);
  stream_.synchronize();

  // Check if conversion yielded a nullptr
  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  // Update completedRows_.
  completedRows_ += output->size();

  // TODO: Update `completedBytes_` here instead of in `addSplit()`

  return output;
}

void ParquetDataSource::totalScanTimeCalculator(void* userData) {
  TotalScanTimeCallbackData* data =
      static_cast<TotalScanTimeCallbackData*>(userData);

  // Record end time in callback
  auto endTimeUs = getCurrentTimeMicro();

  // Calculate elapsed time in microseconds and convert to nanoseconds
  auto elapsedUs = endTimeUs - data->startTimeUs;
  auto elapsedNs = elapsedUs * 1000; // Convert microseconds to nanoseconds

  // Update totalScanTime
  data->ioStats->incTotalScanTime(elapsedNs);

  delete data;
}

void ParquetDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  // Dynamic cast split to `ParquetConnectorSplit`
  split_ = std::dynamic_pointer_cast<ParquetConnectorSplit>(split);
  VLOG(1) << "Adding split " << split_->toString();

  LOG(INFO) << "Adding split" << split_->toString();

  // Clear columnNames if not empty
  if (not columnNames_.empty()) {
    columnNames_.clear();
  }

  // TODO: `completedBytes_` should be updated in `next()` as we read more and
  // more table bytes
  const auto& filePaths = split_->getCudfSourceInfo().filepaths();
  assert(filePaths.size() == 1);
  /*for (const auto& filePath : filePaths) {
    completedBytes_ += std::filesystem::file_size(filePath);
  }*/

  if (FLAGS_velox_cudf_zrl_reader) {
    // LOG(INFO) << "tableHandle NAME : " << tableHandle_->name();
    auto pos = tableHandle_->name().find(".");
    std::string tableHandle_name = tableHandle_->name().substr(
        pos + 1, tableHandle_->name().length() - (pos + 1));
    // LOG(INFO) << "tableHandle extracted name: " << tableHandle_name;
    pos = filePaths[0].find(tableHandle_name + "_parquet_hack");
    // LOG(INFO) << "pos in filePath = " << pos;
    const std::string data_folderpath = filePaths[0].substr(0, pos);
    // LOG(INFO) << "Data folder path: " << data_folderpath;

    std::regex sf_regex("[0-9]+.parquet");
    auto sf_begin = std::sregex_iterator(
        filePaths[0].begin(), filePaths[0].end(), sf_regex);
    std::string parquet_file_name = (*(sf_begin)).str();
    pos = parquet_file_name.find('.');
    uint32_t chunk = std::stoi(parquet_file_name.substr(0, pos));
    LOG(INFO) << "Pushing chunk : " << chunk;
    chunks_.push_back(chunk);

    if (!splitReaderZrl_) {
      // Create a TableReader
      LOG(INFO) << "About to create TableReader";
      splitReaderZrl_ = createSplitReader(data_folderpath, tableHandle_name);
    }

    // LOG(INFO) << "Adding SPLIT column names";
    for (auto columnName : outputType_->names()) {
      columnNames_.emplace_back(columnName);
    }
  } else {
    // Create a `cudf::io::chunked_parquet_reader` SplitReader
    splitReaderCudf_ = createSplitReader();
  }
}

std::unique_ptr<ibm::velox::cudf_velox::connector::parquet_hack::TableReader>
ParquetDataSource::createSplitReader(
    const std::string& data_folderpath,
    const std::string& tableName) {
  if (subfieldFilterExprSet_) {
    LOG(INFO) << "Got subfieldFilterExprSet";
    auto subfieldFilterExpr = subfieldFilterExprSet_->expr(0);

    // non-ast instructions in filter is not supported for SubFieldFilter.
    // precomputeInstructions which are non-ast instructions should be empty.
    std::vector<PrecomputeInstruction> precomputeInstructions;

    const RowTypePtr readerFilterType_ = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> new_names;
        std::vector<TypePtr> new_types;

        for (const auto& name : readColumnNames_) {
          // Ensure all columns being read are available to the filter
          auto parsedType = tableHandle_->dataColumns()->findChild(name);
          new_names.emplace_back(std::move(name));
          new_types.push_back(parsedType);
        }

        return ROW(std::move(new_names), std::move(new_types));
      } else {
        return outputType_;
      }
    }();

    createAstTree(
        subfieldFilterExpr,
        subfieldTree_,
        subfieldScalars_,
        readerFilterType_,
        precomputeInstructions);
    VELOX_CHECK_EQ(precomputeInstructions.size(), 0);
    // readerOptions.set_filter(subfieldTree_.back());
  }

  std::unique_ptr<ibm::velox::cudf_velox::connector::parquet_hack::TableReader>
      table_reader;
  try {
    table_reader = std::make_unique<
        ibm::velox::cudf_velox::connector::parquet_hack::TableReader>(
        data_folderpath, tableName);
  } catch (const std::runtime_error& e) {
    LOG(INFO) << "Got exception : " << e.what();
  }
  return table_reader;
}

std::unique_ptr<cudf::io::chunked_parquet_reader>
ParquetDataSource::createSplitReader() {
  LOG(INFO) << "Creating Split reader";
  // Reader options
  auto readerOptions =
      cudf::io::parquet_reader_options::builder(split_->getCudfSourceInfo())
          .skip_rows(parquetConfig_->skipRows())
          .use_pandas_metadata(parquetConfig_->isUsePandasMetadata())
          .use_arrow_schema(parquetConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              parquetConfig_->isAllowMismatchedParquetSchemas())
          .timestamp_type(parquetConfig_->timestampType())
          .build();

  // Set num_rows only if available
  if (parquetConfig_->numRows().has_value()) {
    LOG(INFO) << "Setting num rows";
    readerOptions.set_num_rows(parquetConfig_->numRows().value());
  }

  if (subfieldFilterExprSet_) {
    LOG(INFO) << "Got subfieldFilterExprSet";
    auto subfieldFilterExpr = subfieldFilterExprSet_->expr(0);

    // non-ast instructions in filter is not supported for SubFieldFilter.
    // precomputeInstructions which are non-ast instructions should be empty.
    std::vector<PrecomputeInstruction> precomputeInstructions;

    const RowTypePtr readerFilterType_ = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> new_names;
        std::vector<TypePtr> new_types;

        for (const auto& name : readColumnNames_) {
          // Ensure all columns being read are available to the filter
          auto parsedType = tableHandle_->dataColumns()->findChild(name);
          new_names.emplace_back(std::move(name));
          new_types.push_back(parsedType);
        }

        return ROW(std::move(new_names), std::move(new_types));
      } else {
        return outputType_;
      }
    }();

    createAstTree(
        subfieldFilterExpr,
        subfieldTree_,
        subfieldScalars_,
        readerFilterType_,
        precomputeInstructions);
    VELOX_CHECK_EQ(precomputeInstructions.size(), 0);
    readerOptions.set_filter(subfieldTree_.back());
  }

  // Set column projection if needed
  if (readColumnNames_.size()) {
    readerOptions.set_columns(readColumnNames_);
  }

  stream_ = cudfGlobalStreamPool().get_stream();
  // Create a parquet reader
  return std::make_unique<cudf::io::chunked_parquet_reader>(
      parquetConfig_->maxChunkReadLimit(),
      parquetConfig_->maxPassReadLimit(),
      readerOptions,
      stream_,
      cudf::get_current_device_resource_ref());
}

void ParquetDataSource::resetSplit() {
  LOG(INFO) << "Resetting split";
  split_.reset();
  if (FLAGS_velox_cudf_zrl_reader)
    splitReaderZrl_.reset();
  else
    splitReaderCudf_.reset();
  columnNames_.clear();
}

std::unordered_map<std::string, RuntimeCounter>
ParquetDataSource::runtimeStats() {
  auto res = runtimeStats_.toMap();
  res.insert({
      {"totalScanTime",
       RuntimeCounter(ioStats_->totalScanTime(), RuntimeCounter::Unit::kNanos)},
      {"totalRemainingFilterTime",
       RuntimeCounter(
           totalRemainingFilterTime_.load(std::memory_order_relaxed),
           RuntimeCounter::Unit::kNanos)},
  });
  return res;
}

} // namespace facebook::velox::cudf_velox::connector::parquet
