#include "velox/experimental/cudf/connectors/parquet/TableReader.hpp"
#include "velox/experimental/cudf/connectors/parquet/error.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <kvikio/file_handle.hpp>

#include <regex>
#include <stdexcept>

namespace ibm::velox::cudf_velox::connector::parquet_hack {

ColumnType ConvertStringToColumnType(const std::string& s) {
  static const std::map<std::string, ColumnType> algo_map = {
      {"char", ColumnType::Char},
      {"short", ColumnType::Int16},
      {"int", ColumnType::Int32},
      {"long", ColumnType::Int64},
      {"float", ColumnType::Float32},
      {"double", ColumnType::Float64},
      {"string", ColumnType::String},
      {"offset", ColumnType::Offset}};
  auto it = algo_map.find(s);
  if (it != algo_map.end()) {
    return it->second;
  } else {
    return ColumnType::Unspecified;
  }
}

/// @brief Returns the index of to the named column given the list of columns.
/// @param col_name The name of the column
/// @param columns The list of the columns that are present in the table.
/// @return A column index.
int32_t get_col_idx(
    std::string col_name,
    std::vector<std::string> const& columns) {
  return std::distance(
      columns.begin(), std::find(columns.begin(), columns.end(), col_name));
}

static int32_t find_column_index(
    const std::vector<std::string>& columns,
    const std::string& col_name) {
  auto it = std::find(columns.begin(), columns.end(), col_name);

  if (it != columns.end()) {
    return std::distance(columns.begin(), it);
  } else {
    return -1;
  }
}

/**
 * @brief Converts a short into a TIMESTAMP
 *
 * @param col The column to convert
 * @param stream The CUDA stream used for device memory operations and kernel
 * launches.
 * @param mr Device memory resource used to allocate the returned column's
 * device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> short_to_timestamp(
    cudf::column_view const& col,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::cast(
      cudf::cast(col, cudf::data_type{cudf::type_id::DURATION_DAYS}, stream, mr)
          ->view(),
      cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
      stream,
      mr);
}

/**
 * @brief Converts a short into a double without changing the value.alignas
 *
 * @param col The column to convert
 * @param stream The CUDA stream used for device memory operations and kernel
 * launches.
 * @param mr Device memory resource used to allocate the returned column's
 * device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> short_to_double(
    cudf::column_view const& col,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::cast(col, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr);
}

/**
 * @brief Converts a fixpoint value with 2 decimal digits (think: cents)
 * into a double (think dollars and fractions thereof.)
 */
[[nodiscard]] std::unique_ptr<cudf::column> fixpoint_to_double(
    cudf::column_view const& col,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const hundred = cudf::numeric_scalar<double>(100.0);
  auto const double_type = cudf::data_type{cudf::type_id::FLOAT64};
  return cudf::binary_operation(
      col, hundred, cudf::binary_operator::DIV, double_type, stream, mr);
}

/**
 * @brief Converts a short into an int
 *
 * @param col The column to convert
 * @param stream The CUDA stream used for device memory operations and kernel
 * launches.
 * @param mr Device memory resource used to allocate the returned column's
 * device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> short_to_int(
    cudf::column_view const& col,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::cast(col, cudf::data_type{cudf::type_id::INT32}, stream, mr);
}

/**
 * @brief Converts a short into a 64 bit int
 *
 * @param col The column to convert
 * @param stream The CUDA stream used for device memory operations and kernel
 * launches.
 * @param mr Device memory resource used to allocate the returned column's
 * device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> short_to_bigint(
    cudf::column_view const& col,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::cast(col, cudf::data_type{cudf::type_id::INT64}, stream, mr);
}

TableReader::TableReader(
    const std::string& base_dir,
    const std::string& table_name)
    : base_dir_(base_dir), table_name_(table_name) {
  Init();
}

std::unique_ptr<cudf::column> TableReader::LoadColumnChunk(
    const std::string& col_name,
    const uint32_t chunk,
    rmm::cuda_stream_view stream) {
  size_t bytes_read = 0;
  return LoadColumnChunkAndGetBytes(col_name, chunk, bytes_read, stream);
}

std::unique_ptr<cudf::column> TableReader::LoadColumnChunkAndGetBytes(
    const std::string& col_name,
    uint32_t chunk,
    size_t& read_bytes,
    rmm::cuda_stream_view stream) {
  // First, check whether the column exists.
  const auto col_meta_ptr = columns_.find(col_name);
  if (col_meta_ptr == columns_.end()) {
    throw new std::invalid_argument("Column doesn't exist: " + col_name);
  }
  if (chunk <= 0 || chunk > num_chunks()) {
    throw new std::invalid_argument(
        "Chunk doesn't exist: " + std::to_string(chunk));
  }
  --chunk; // change to index from 0 up to but not including num_slices.

  // Load the column using Kvikio.
  std::unique_ptr<rmm::device_buffer> data_buf;
  std::unique_ptr<cudf::column> result_column;
  size_t tmp_read_bytes{0};
  data_buf = ReadFileKvikio(
      col_meta_ptr->second.chunks[chunk], tmp_read_bytes, stream);
  read_bytes = read_bytes + tmp_read_bytes;
  if (col_meta_ptr->second.type == ColumnType::String) {
    // also load the offset table.
    std::unique_ptr<rmm::device_buffer> offset_buf;
    std::string offset_name =
        std::string("offset") + col_meta_ptr->second.column_name;
    const auto offset_meta_ptr = columns_.find(offset_name);
    if (offset_meta_ptr == columns_.end()) {
      throw new std::invalid_argument(
          "Offset column for string column doesn't exist: " + col_name);
    }
    offset_buf = ReadFileKvikio(
        offset_meta_ptr->second.chunks[chunk], tmp_read_bytes, stream);
    read_bytes = read_bytes + tmp_read_bytes;
    // create the cudf::column.
    // TODO: Assumption here is that we don't have "long columns" where
    // cudf::size_type (an int32) is not

    std::size_t num_offsets = offset_buf->size() / sizeof(cudf::size_type);
    std::unique_ptr<cudf::column> offsets_column =
        std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_to_id<cudf::size_type>()},
            num_offsets,
            std::move(*offset_buf.release()),
            rmm::device_buffer{},
            0);

    result_column = cudf::make_strings_column(
        num_offsets - 1,
        std::move(offsets_column),
        std::move(*data_buf.release()),
        0,
        rmm::device_buffer{});
  } else {
    std::size_t num_elems;
    switch (col_meta_ptr->second.type) {
      case ColumnType::Char:
        num_elems = data_buf->size();
        /*result_column =
           std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<char>()},
                                                       num_elems,
                                                       std::move(*data_buf.release()),
                                                       rmm::device_buffer{},
                                                       0);*/
        result_column = cudf::make_strings_column(
            num_elems,
            std::move(
                cudf::sequence(
                    num_elems + 1, cudf::numeric_scalar<int32_t>(0), stream)),
            std::move(*data_buf.release()),
            0,
            rmm::device_buffer{});
        break;
      case ColumnType::Int16:
        num_elems = data_buf->size() / sizeof(int16_t);
        result_column = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_to_id<int16_t>()},
            num_elems,
            std::move(*data_buf.release()),
            rmm::device_buffer{},
            0);
        if (col_meta_ptr->second.column_name.find("date") !=
            std::string::npos) {
          result_column = short_to_timestamp(
              result_column->view(),
              stream,
              cudf::get_current_device_resource_ref());
        } else if (
            col_meta_ptr->second.column_name == "tax" ||
            col_meta_ptr->second.column_name == "discount") {
          result_column = fixpoint_to_double(
              result_column->view(),
              stream,
              cudf::get_current_device_resource_ref());
        } else if (
            col_meta_ptr->second.column_name == "quantity" ||
            col_meta_ptr->second.column_name == "totalprice") {
          result_column = short_to_double(
              result_column->view(),
              stream,
              cudf::get_current_device_resource_ref());
        } else if (
            col_meta_ptr->second.column_name == "shippriority" ||
            col_meta_ptr->second.column_name == "size") {
          result_column = short_to_int(
              result_column->view(),
              stream,
              cudf::get_current_device_resource_ref());
        } else if (
            col_meta_ptr->second.column_name == "nationkey" ||
            col_meta_ptr->second.column_name == "regionkey") {
          result_column = short_to_bigint(
              result_column->view(),
              stream,
              cudf::get_current_device_resource_ref());
        }
        break;
      case ColumnType::Int32:
        num_elems = data_buf->size() / sizeof(int32_t);
        result_column = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_to_id<int32_t>()},
            num_elems,
            std::move(*data_buf.release()),
            rmm::device_buffer{},
            0);
        if (col_meta_ptr->second.column_name == "extendedprice" ||
            col_meta_ptr->second.column_name == "retailprice" ||
            col_meta_ptr->second.column_name == "supplycost") {
          result_column = fixpoint_to_double(
              result_column->view(),
              stream,
              cudf::get_current_device_resource_ref());
        } else if (col_meta_ptr->second.column_name == "acctbal") {
          result_column = short_to_double(
              result_column->view(),
              stream,
              cudf::get_current_device_resource_ref());
        }
        break;
      case ColumnType::Int64:
        num_elems = data_buf->size() / sizeof(int64_t);
        result_column = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_to_id<int64_t>()},
            num_elems,
            std::move(*data_buf.release()),
            rmm::device_buffer{},
            0);
        break;
      case ColumnType::Float32:
        num_elems = data_buf->size() / sizeof(float);
        result_column = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_to_id<float>()},
            num_elems,
            std::move(*data_buf.release()),
            rmm::device_buffer{},
            0);
        break;
      case ColumnType::Float64:
        num_elems = data_buf->size() / sizeof(double);
        result_column = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_to_id<double>()},
            num_elems,
            std::move(*data_buf.release()),
            rmm::device_buffer{},
            0);
        break;
      default:
        throw new std::runtime_error(
            "Unknown column type for column: " + col_name);
        break;
    }
  }
  return result_column;
}

std::unique_ptr<cudf::table> TableReader::LoadTableChunk(
    const std::vector<std::string>& column_names,
    uint32_t chunk,
    rmm::cuda_stream_view stream) {
  size_t read_bytes{0};
  return LoadTableChunkAndGetBytes(column_names, chunk, read_bytes, stream);
}

/**
 * Loads the given columns and returns a cudf::table object.alignas
 */
std::unique_ptr<cudf::table> TableReader::LoadTableChunkAndGetBytes(
    const std::vector<std::string>& column_names,
    uint32_t chunk,
    size_t& read_bytes,
    rmm::cuda_stream_view stream) {
  // First, check whether the columns exist.
  for (const auto& col : column_names) {
    if (columns_.find(col) == columns_.end()) {
      throw new std::invalid_argument("Column doesn't exist: " + col);
    }
  }
  if (chunk <= 0 || chunk > num_chunks()) {
    throw new std::invalid_argument(
        "Chunk doesn't exist: " + std::to_string(chunk));
  }
  std::vector<std::thread> threads;
  threads.reserve(column_names.size());
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<std::future<std::unique_ptr<cudf::column>>> futures;
  futures.reserve(column_names.size());
  int current_device{0};
  CUDA_RT_CALL(cudaGetDevice(&current_device));

  std::atomic<size_t> total_bytes_read{0};
  for (const auto& col : column_names) {
    std::packaged_task<std::unique_ptr<cudf::column>()> loader(
        [col, chunk, &total_bytes_read, stream, current_device, this]() {
          CUDA_RT_CALL(cudaSetDevice(current_device));
          std::unique_ptr<cudf::column> column;
          size_t tmp_bytes_read{0};
          column = this->LoadColumnChunkAndGetBytes(
              col, chunk, tmp_bytes_read, stream);
          total_bytes_read.fetch_add(tmp_bytes_read, std::memory_order_relaxed);
          return column;
        });
    futures.emplace_back(loader.get_future());
    // launch the loader on a thread.
    threads.emplace_back(std::move(loader));
  }
  // wait for all the future and collect the results.
  columns.clear();
  for (auto& f : futures) {
    columns.emplace_back(std::move(f.get()));
  }
  for (auto& t : threads) {
    t.join();
  }
  // create a table from the columns.
  read_bytes = total_bytes_read.load();
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> TableReader::LoadTableSeq(
    const std::vector<std::string>& all_column_names,
    const std::vector<std::string>& column_names,
    uint32_t chunk,
    std::optional<std::reference_wrapper<cudf::ast::expression const>> filter,
    std::vector<LikeExpr>& like_exprs,
    size_t& read_bytes,
    rmm::cuda_stream_view stream) {
  std::unique_ptr<cudf::table> tmp_table, filtered_table,
      result_table = nullptr;

  // column_names must be a subset of all_column_names.
  for (const auto& col : column_names) {
    if (std::find(all_column_names.begin(), all_column_names.end(), col) ==
        all_column_names.end()) {
      std::cout << "Column doesn't exist in all_column_names: " << col
                << std::endl;
      throw new std::runtime_error(
          "Column doesn't exist in all_column_names: " + col);
    }
  }
  // loop over all chunks.
  // while ((chunk = chunk_num_gen()) > 0) {
  std::unique_ptr<cudf::column> filter_mask;
  tmp_table =
      LoadTableChunkAndGetBytes(all_column_names, chunk, read_bytes, stream);
  if (filter.has_value()) {
    // Apply the filter.
    filter_mask =
        cudf::compute_column(tmp_table->view(), filter.value(), stream);
  }
  // apply like expression.
  for (LikeExpr& like_expr : like_exprs) {
    cudf::strings_column_view str_col =
        tmp_table->view().column(like_expr.column_idx);
    std::unique_ptr<cudf::column> like_mask;
    cudf::string_scalar like_pattern(like_expr.like_pattern, true, stream);
    cudf::string_scalar escape_pattern("", true, stream);
    like_mask =
        cudf::strings::like(str_col, like_pattern, escape_pattern, stream);
    // build the logical AND expression of the filter mask and the like mask.
    if (filter_mask != nullptr) {
      auto const bool_type = cudf::data_type{cudf::type_id::BOOL8};
      auto tmp_col = cudf::binary_operation(
          filter_mask->view(),
          like_mask->view(),
          cudf::binary_operator::LOGICAL_AND,
          bool_type,
          stream);
      filter_mask = std::move(tmp_col);
    } else {
      filter_mask = std::move(like_mask);
    }
  }
  if (filter_mask != nullptr) {
    filtered_table = cudf::apply_boolean_mask(
        tmp_table->view(), filter_mask->view(), stream);
  }
  if (filtered_table == nullptr) {
    // no filter. The filtered table is the same as the tmp table.
    filtered_table = std::move(tmp_table);
  }
  if (result_table == nullptr) {
    result_table = std::move(filtered_table);
  } else {
    std::vector<cudf::table_view> table_views = {
        result_table->view(), filtered_table->view()};
    result_table = cudf::concatenate(
        cudf::host_span<cudf::table_view const>(
            table_views.data(), table_views.size()),
        stream);
    filtered_table = nullptr;
  }
  //}
  if (column_names.size() < all_column_names.size()) {
    // compute the projection of column_names.
    auto cols = result_table->release();
    std::vector<std::unique_ptr<cudf::column>> projected_cols;
    for (const auto& col : column_names) {
      auto idx = find_column_index(all_column_names, col);
      if (idx >= 0) {
        projected_cols.push_back(std::move(cols[idx]));
      }
    }
    result_table = std::make_unique<cudf::table>(std::move(projected_cols));
  }
  return result_table;
}

std::size_t TableReader::CountRows(const std::string& col_name) {
  std::size_t num_rows;

  // First, check whether the column exists.
  const auto col_meta_ptr = columns_.find(col_name);
  if (col_meta_ptr == columns_.end()) {
    throw new std::runtime_error("Column doesn't exist: " + col_name);
  }
  std::size_t byte_counter = 0L;
  for (uint32_t chunk = 0; chunk < num_chunks(); ++chunk) {
    auto file = col_meta_ptr->second.chunks[chunk];
    kvikio::FileHandle kvik_file(file.c_str(), "r");
    byte_counter += kvik_file.nbytes(); // get the file size.
  }
  switch (col_meta_ptr->second.type) {
    case ColumnType::Char:
      num_rows = byte_counter;
      break;
    case ColumnType::Int16:
      num_rows = byte_counter / sizeof(int16_t);
      break;
    case ColumnType::Int32:
      num_rows = byte_counter / sizeof(int32_t);
      break;
    case ColumnType::Int64:
      num_rows = byte_counter / sizeof(int64_t);
      break;
    default:
      throw new std::runtime_error("Column type not supported: " + col_name);
      break;
  }
  return num_rows;
}

// Populates the table's metadata by reading the directory.
void TableReader::Init() {
  if (columns_.empty()) {
    LOG(INFO) << "TableReader::Init()";
    std::vector<std::filesystem::path> paths;
    EnumerateReadableFiles(paths);
    // LOG(INFO) << "Got paths";
    //  extract the column type from the file name.
    //  The assumption here is that '-' is not part of the table name and not
    //  part of the column name.
    //  Example: lineitem-tax-double.ans.81
    std::regex pattern(".*-(.*?)-(.*?)\\.(.*?)\\.(.*?)");
    std::smatch match;
    auto iter = paths.begin();
    while (iter != paths.end()) {
      std::string filename = iter->filename().string();
      // LOG(INFO) << "Got file: " << iter->filename().string();
      ColumnMeta meta;
      if (std::regex_match(filename, match, pattern)) {
        meta.column_name = match[1].str();
        meta.type = ConvertStringToColumnType(match[2].str());
        std::string col_tab_type =
            table_name_ + "-" + meta.column_name + "-" + match[2].str();
        if (meta.type == ColumnType::Unspecified) {
          throw new std::runtime_error(
              "Unknown column type in filename: " + filename);
        }
        // put all the files belonging to the same column into the
        // chunk vector. Since the vector is sorted, all the columns
        // are adjacent.
        do {
          meta.chunks.push_back(*iter);
          ++iter;
        } while (iter != paths.end() &&
                 iter->filename().string().rfind(col_tab_type, 0) == 0);
        // handle offsets differently.
        if (meta.type == ColumnType::Offset) {
          std::string offset_name = std::string("offset") + meta.column_name;
          columns_.insert({offset_name, meta});
        } else {
          columns_.insert({meta.column_name, meta});
        }
      } else {
        LOG(INFO) << "Couldn't extract metadata from filename: " + filename;
        throw new std::runtime_error(
            "Couldn't extract metadata from filename: " + filename);
      }
    }
  }
}

// Populates the vector of paths.
void TableReader::EnumerateReadableFiles(
    std::vector<std::filesystem::path>& data_files) {
  std::filesystem::path table_dir =
      std::filesystem::path(base_dir_) / std::filesystem::path(table_name_);
  data_files.clear();
  if (!std::filesystem::exists(table_dir)) {
    LOG(INFO) << "Table directory doesn't exist: " << table_dir.string();
    throw new std::runtime_error(
        "Table directory doesn't exist: " + table_dir.string());
  }
  for (const auto& entry : std::filesystem::directory_iterator(table_dir)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      // check whether table_name_ is a prefix of filename by limiting the
      // reverse find to the first position.
      if (filename.rfind(table_name_, 0) == 0) {
        // get the files matching the column.
        data_files.push_back(entry.path());
      }
    }
  }
  if (data_files.empty()) {
    throw new std::runtime_error("No files found in " + table_dir.string());
  }
  std::sort(data_files.begin(), data_files.end());
}

std::unique_ptr<rmm::device_buffer> TableReader::ReadFileKvikio(
    std::filesystem::path file,
    size_t& read_bytes,
    rmm::cuda_stream_view stream) const {
  const std::size_t file_offset = 0L;
  std::string e_msg;
  std::future<std::size_t> read_future;
  std::unique_ptr<rmm::device_buffer> dev_buf_out;
  kvikio::FileHandle kvik_file(file.c_str(), "r");

  std::size_t nbytes = kvik_file.nbytes(); // get the file size.
                                           // allocate device buffer.
  dev_buf_out = std::make_unique<rmm::device_buffer>(nbytes, stream);
  read_future = kvik_file.pread(dev_buf_out->data(), nbytes, file_offset);
  if (!read_future.valid()) {
    // something bad happend.
    throw new std::runtime_error(
        "Received invalid future from kvikio file read.");
  }
  read_bytes = read_future.get(); // wait for the future to complete.
  if (read_bytes != nbytes) {
    throw new std::runtime_error(
        "Received invalid number of bytes from kvikio file read.");
  }

  kvik_file.close();
  return dev_buf_out;
}
} // namespace ibm::velox::cudf_velox::connector::parquet_hack
