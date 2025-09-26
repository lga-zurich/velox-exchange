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

#pragma once

#include "velox/experimental/cudf-exchange/CudfExchangeClient.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"

#include <gflags/gflags.h>

DECLARE_bool(velox_cudf_enabled);
DECLARE_string(velox_cudf_memory_resource);
DECLARE_bool(velox_cudf_debug);
DECLARE_bool(velox_cudf_exchange);

namespace facebook::velox::cudf_velox {

static const std::string kCudfAdapterName = "cuDF";

// QueryConfig key. Enable or disable cudf in task level.
static const std::string kCudfEnabled = "cudf.enabled";

struct TaskPlanNodeKey {
  std::string taskId;
  core::PlanNodeId planNodeId;

  TaskPlanNodeKey(const std::string& tid, const core::PlanNodeId& pid)
      : taskId(tid), planNodeId(pid) {}

  // need equality operator for unordered map.
  bool operator==(const TaskPlanNodeKey& other) const {
    return taskId == other.taskId && planNodeId == other.planNodeId;
  }

  // Need a hash functor for the unordered map.
  struct Hash {
    std::size_t operator()(const TaskPlanNodeKey& key) const {
      std::hash<std::string> hasher;
      std::size_t h1 = hasher(key.taskId);
      std::size_t h2 = hasher(key.planNodeId);
      return h1 ^ (h2 << 1); // simple combination of the two hash functions.
    }
  };
};

static std::unordered_map<
  TaskPlanNodeKey,
  std::shared_ptr<cudf_exchange::CudfExchangeClient>,
  TaskPlanNodeKey::Hash> cudfExchangeClientByTaskAndPlanNode_;

class CompileState {
 public:
  CompileState(const exec::DriverFactory& driverFactory, exec::Driver& driver)
      : driverFactory_(driverFactory), driver_(driver) {}

  exec::Driver& driver() {
    return driver_;
  }

  // Replaces sequences of Operators in the Driver given at construction with
  // cuDF equivalents. Returns true if the Driver was changed.
  bool compile(bool force_replace);

  std::shared_ptr<cudf_exchange::CudfExchangeClient> createCudfExchangeClient(
      const std::string& taskId,
      const core::PlanNodeId& planNodeId,
      const int destination,
      const int32_t numberOfConsumers,
      folly::Executor* executor);

  const exec::DriverFactory& driverFactory_;
  exec::Driver& driver_;
};

class CudfOptions {
 public:
  static CudfOptions& getInstance() {
    static CudfOptions instance;
    return instance;
  }

  void setPrefix(const std::string& prefix) {
    prefix_ = prefix;
  }

  const std::string& prefix() const {
    return prefix_;
  }

  void setShouldTransformLastOutput(bool newValue) {
    transformLastOutput_ = newValue;
  }

  const bool shouldTransformLastOutput() const {
    return transformLastOutput_;
  }

  const bool cudfEnabled;
  const std::string cudfMemoryResource;
  const bool cudfExchange;
  // The initial percent of GPU memory to allocate for memory resource for one
  // thread.
  int memoryPercent;
  const bool force_replace;

  CudfOptions(bool force_repl)
      : cudfEnabled(FLAGS_velox_cudf_enabled),
        cudfMemoryResource(FLAGS_velox_cudf_memory_resource),
        cudfExchange(FLAGS_velox_cudf_exchange),
        memoryPercent(50),
        force_replace{force_repl},
        prefix_("") {}

 private:
  CudfOptions()
      : cudfEnabled(FLAGS_velox_cudf_enabled),
        cudfMemoryResource(FLAGS_velox_cudf_memory_resource),
        cudfExchange(FLAGS_velox_cudf_exchange),
        memoryPercent(50),
        force_replace{false},
        prefix_(""),
        transformLastOutput_(false) {}
  CudfOptions(const CudfOptions&) = delete;
  CudfOptions& operator=(const CudfOptions&) = delete;
  std::string prefix_;
  bool transformLastOutput_;
};

/// Registers adapter to add cuDF operators to Drivers.
void registerCudf(const CudfOptions& options = CudfOptions::getInstance());
void unregisterCudf();

/// Returns true if cuDF is registered.
bool cudfIsRegistered();

/**
 * @brief Returns true if the velox_cudf_debug flag is set to true.
 */
bool cudfDebugEnabled();

} // namespace facebook::velox::cudf_velox
