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

#include "velox/connectors/hive/storage_adapters/s3fs/S3ReadFile.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Counters.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"

#include <aws/core/Aws.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/transfer/TransferManager.h>
#include <aws/transfer/TransferHandle.h>


namespace facebook::velox::filesystems {

namespace {

// By default, the AWS SDK reads object data into a backed Stream.
// To avoid copies, it read directly into a pre-allocated buffer instead.
// See https://github.com/aws/aws-sdk-cpp/issues/64 for an alternative but
// functionally similar recipe.
class UnderlyingStreamWrapper : public Aws::IOStream {
  public:
    explicit UnderlyingStreamWrapper(std::streambuf* buf) : Aws::IOStream(buf) {}
    ~UnderlyingStreamWrapper() override = default;
};

} // namespace

class S3ReadFile ::Impl {
 public:
  explicit Impl(std::string_view path, std::shared_ptr<Aws::S3::S3Client> client, std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor> executor)
      : client_(client), executor_(executor) {
    getBucketAndKeyFromPath(path, bucket_, key_);
  }

  // Gets the length of the file.
  // Checks if there are any issues reading the file.
  void initialize(const filesystems::FileOptions& options) {
    if (options.fileSize.has_value()) {
      VELOX_CHECK_GE(
          options.fileSize.value(), 0, "File size must be non-negative");
      length_ = options.fileSize.value();
    }

    // Make it a no-op if invoked twice.
    if (length_ != -1) {
      return;
    }

    Aws::S3::Model::HeadObjectRequest request;
    request.SetBucket(awsString(bucket_));
    request.SetKey(awsString(key_));

    RECORD_METRIC_VALUE(kMetricS3MetadataCalls);
    auto outcome = client_->HeadObject(request);
    if (!outcome.IsSuccess()) {
      RECORD_METRIC_VALUE(kMetricS3GetMetadataErrors);
    }
    RECORD_METRIC_VALUE(kMetricS3GetMetadataRetries, outcome.GetRetryCount());
    VELOX_CHECK_AWS_OUTCOME(
        outcome, "Failed to get metadata for S3 object", bucket_, key_);
    length_ = outcome.GetResult().GetContentLength();
    VELOX_CHECK_GE(length_, 0);
    Aws::Transfer::TransferManagerConfiguration transferConfig(executor_.get());
    transferConfig.s3Client = client_;
    transferManager_ = Aws::Transfer::TransferManager::Create(transferConfig);
  }

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buffer,
      const FileStorageContext& fileStorageContext) const {
    preadInternal(offset, length, static_cast<char*>(buffer));
    return {static_cast<char*>(buffer), length};
  }

  std::string pread(
      uint64_t offset,
      uint64_t length,
      const FileStorageContext& fileStorageContext) const {
    std::string result(length, 0);
    char* position = result.data();
    preadInternal(offset, length, position);
    return result;
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileStorageContext& fileStorageContext) const {
    // 'buffers' contains Ranges(data, size)  with some gaps (data = nullptr) in
    // between. This call must populate the ranges (except gap ranges)
    // sequentially starting from 'offset'. AWS S3 GetObject does not support
    // multi-range. AWS S3 also charges by number of read requests and not size.
    // The idea here is to use a single read spanning all the ranges and then
    // populate individual ranges. We pre-allocate a buffer to support this.
    size_t length = 0;
    for (const auto range : buffers) {
      length += range.size();
    }
    // TODO: allocate from a memory pool
    std::string result(length, 0);
    preadInternal(offset, length, static_cast<char*>(result.data()));
    size_t resultOffset = 0;
    for (auto range : buffers) {
      if (range.data()) {
        memcpy(range.data(), &(result.data()[resultOffset]), range.size());
      }
      resultOffset += range.size();
    }
    return length;
  }

  uint64_t size() const {
    return length_;
  }

  uint64_t memoryUsage() const {
    // TODO: Check if any buffers are being used by the S3 library
    return sizeof(Aws::S3::S3Client) + kS3MaxKeySize + 2 * sizeof(std::string) +
        sizeof(int64_t);
  }

  bool shouldCoalesce() const {
    return false;
  }

  std::string getName() const {
    return fmt::format("s3://{}/{}", bucket_, key_);
  }

 private:
  // The assumption here is that "position" has space for at least "length"
  // bytes.
  void preadInternal(uint64_t offset, uint64_t length, char* position) const {
    // Read the desired range of bytes.
  Aws::Utils::Stream::PreallocatedStreamBuf streamBuffer(
        (unsigned char*)position, length);
    auto downloadHandle =
        transferManager_->DownloadFile(bucket_, key_, offset, length, [&]() {
          return Aws::New<UnderlyingStreamWrapper>("TestTag", &streamBuffer);
        });
    int count = 1;
    using namespace std::chrono;
    uint64_t startTime =
        duration_cast<milliseconds>(system_clock::now().time_since_epoch())
            .count();
    uint64_t now = startTime;
    uint64_t logAt = startTime + 5000; // wait 5 seconds; then warn and backoff
    auto status = downloadHandle->GetStatus();
    while (status == Aws::Transfer::TransferStatus::NOT_STARTED ||
           status == Aws::Transfer::TransferStatus::IN_PROGRESS) {
      status = downloadHandle->GetStatus();
      now = duration_cast<milliseconds>(system_clock::now().time_since_epoch())
                .count();
      if (now > logAt) {
        VLOG(1) << "S3 Request is taking a long time: "
                << (now - startTime) / 1000 << "s!\n";
        count++;
        logAt += count * 5000; // backoff for next warning
      }
    }
    VELOX_CHECK(
        downloadHandle->GetStatus() == Aws::Transfer::TransferStatus::COMPLETED,
        "Failed to get S3 object using the transfer manager from location: {}:{}",
        bucket_,
        key_);
  }

  std::shared_ptr<Aws::S3::S3Client> client_;
  std::shared_ptr<Aws::Transfer::TransferManager> transferManager_;
  std::string bucket_;
  std::string key_;
  int64_t length_ = -1;
  std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor> executor_;
};

S3ReadFile::S3ReadFile(std::string_view path, std::shared_ptr<Aws::S3::S3Client> client,
      std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor>
          executor) {
  impl_ = std::make_shared<Impl>(path, client, executor);
}

S3ReadFile::~S3ReadFile() = default;

void S3ReadFile::initialize(const filesystems::FileOptions& options) {
  return impl_->initialize(options);
}

std::string_view S3ReadFile::pread(
    uint64_t offset,
    uint64_t length,
    void* buf,
    const FileStorageContext& fileStorageContext) const {
  return impl_->pread(offset, length, buf, fileStorageContext);
}

std::string S3ReadFile::pread(
    uint64_t offset,
    uint64_t length,
    const FileStorageContext& fileStorageContext) const {
  return impl_->pread(offset, length, fileStorageContext);
}

uint64_t S3ReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers,
    const FileStorageContext& fileStorageContext) const {
  return impl_->preadv(offset, buffers, fileStorageContext);
}

uint64_t S3ReadFile::size() const {
  return impl_->size();
}

uint64_t S3ReadFile::memoryUsage() const {
  return impl_->memoryUsage();
}

bool S3ReadFile::shouldCoalesce() const {
  return impl_->shouldCoalesce();
}

std::string S3ReadFile::getName() const {
  return impl_->getName();
}

} // namespace facebook::velox::filesystems
