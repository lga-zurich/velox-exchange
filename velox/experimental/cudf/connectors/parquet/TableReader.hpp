#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <filesystem>
#include <functional>
#include <rmm/cuda_stream_view.hpp>

#include <glog/logging.h>

namespace ibm::velox::cudf_velox::connector::parquet_hack {

enum class ColumnType : char {
    Char = 0,
    Int16 = 1,
    Int32 = 2,
    Int64 = 3,
    Float32 = 4,
    Float64 = 5,
    String = 6,
    Offset = 7,  // Offset, an Int32, needs to be treated separately.
    Unspecified = 99
};

// The like expression is used to filter the table by matching the elements
// of a given column with the like pattern. As in SQL like, the '%' stands for
// any number of charaters and '_' stands for a single character.
struct LikeExpr {
    int32_t column_idx;
    std::string like_pattern;
};

/**
 * Class for loading table chunks into cudf device columns.
 */

class TableReader {
   public:
    /**
     * Constructs a new table reader for a given table located in a given
     * directory.
     * @param base_dir The base directory that contains the tables.
     * @param table_name The name of the table
     */
    TableReader(const std::string &base_dir, const std::string &table_name);

    /**
     * Disable default constructor.
     */
    TableReader() = delete;

    /**
     * @brief Loads a chunk of the given column into device memory and returns a cudf::column.
     * @param col_name The column name to load.
     * @param chunk The chunk within then column.
     * @param stream The stream used to do the loading.
     * @return The loaded column .
     */
    [[nodiscard]] std::unique_ptr<cudf::column> LoadColumnChunk(const std::string &col_name,
                                                                uint32_t chunk,
                                                                rmm::cuda_stream_view stream);

    [[nodiscard]] std::unique_ptr<cudf::column> LoadColumnChunkAndGetBytes(const std::string &col_name,
                                                                           uint32_t chunk,
                                                                           size_t &read_bytes,
                                                                           rmm::cuda_stream_view stream);

    /**
     * Loads a chunk of the given columns and returns a cudf::table
     * @param col_names The list of column names to load.
     * @param chunk The chunk to load. THe same chunk is loaded for all columns.
     * @param stream The stream used to do the loading.
     * @return A unique pointer to the loaded table chunk.
     */
    [[nodiscard]] std::unique_ptr<cudf::table> LoadTableChunk(const std::vector<std::string> &column_names,
                                                              uint32_t chunk,
                                                              rmm::cuda_stream_view stream);

    [[nodiscard]] std::unique_ptr<cudf::table> LoadTableChunkAndGetBytes(const std::vector<std::string> &column_names,
                                                                         uint32_t chunk,
                                                                         size_t &read_bytes,
                                                                         rmm::cuda_stream_view stream);

    /**
     * @brief Loads and filters multiple chunks and returns all the chunks concatenated
     * and filtered as a single cudf::table.
     * @param all_column_names The column names that are loaded. This includes the column
     * names that appear in the filter but are not returned in the result table.
     * @param column_names The column names that are returned in the table.
     * @param chunk_num_gen A generator for the chunks to load. This function must return
     * a valid chunk number or -1 if no more chunks should be loaded. An example of a generator
     * function for loading the even chunks between 1 and 10 is:
     *  std::function<int32_t()> chunk_gen = [i = 2]() mutable {
     *      i += 2;
     *      return (i - 2 < 10) ? i - 2 : -1;
     *  };
     * @param filter A filter expression that returns a boolean value. The filter must only use
     * values from columns that are present in all_column_names. The filter is optional. In order
     * to prevent a copy of the filter expression, a reference wrapper is used.
     * @return The loaded, filtered and projjected table.
     */
    [[nodiscard]] std::unique_ptr<cudf::table> LoadTableSeq(
        const std::vector<std::string> &all_column_names,
        const std::vector<std::string> &column_names,
        std::function<int32_t()> chunk_num_gen,
        std::optional<std::reference_wrapper<cudf::ast::expression const>> filter,
        rmm::cuda_stream_view stream);

    /**
     * @brief Loads and filters multiple chunks and returns all the chunks concatenated
     * and filtered as a single cudf::table.
     * @param all_column_names The column names that are loaded. This includes the column
     * names that appear in the filter but are not returned in the result table.
     * @param column_names The column names that are returned in the table.
     * @param chunk_num_gen A generator for the chunks to load. This function must return
     * a valid chunk number or -1 if no more chunks should be loaded. An example of a generator
     * function for loading the even chunks between 1 and 10 is:
     *  std::function<int32_t()> chunk_gen = [i = 2]() mutable {
     *      i += 2;
     *      return (i - 2 < 10) ? i - 2 : -1;
     *  };
     * @param filter A filter expression that returns a boolean value. The filter must only use
     * values from columns that are present in all_column_names. The filter is optional. In order
     * to prevent a copy of the filter expression, a reference wrapper is used.
     * @param like_exprs A vector of SQL "like" expressions for filtering on string columns.
     * The like expressions identifies the column by index and the expression uses '%' for
     * zero or more characters and '_' for a single charater. The column must be present in all_column_names.
     * @return The loaded, filtered and projjected table.
     */
    [[nodiscard]] std::unique_ptr<cudf::table> LoadTableSeq(
        const std::vector<std::string> &all_column_names,
        const std::vector<std::string> &column_names,
        uint32_t chunk,
        std::optional<std::reference_wrapper<cudf::ast::expression const>> filter,
        std::vector<LikeExpr> &like_exprs,
        size_t& read_bytes,
        rmm::cuda_stream_view stream);

    /**
     * @brief Loads and filters multiple chunks and returns all the chunks concatenated
     * and filtered as a single cudf::table. The filter is given as a column of values
     * and only those rows are returned that have a match in the filter column.
     * @param column_names The column names that are loaded.
     * @param chunk_num_gen A generator for the chunks to load. This function must return
     * a valid chunk number or -1 if no more chunks should be loaded. An example of a generator
     * function for loading the even chunks between 1 and 10 is:
     *  std::function<int32_t()> chunk_gen = [i = 2]() mutable {
     *      i += 2;
     *      return (i - 2 < 10) ? i - 2 : -1;
     *  };
     * @param filter_column_name The name of the column in the loaded table that is
     * matched against the filter_column. Must be present in column_names.
     * @param filter_table A single-column table with values that are used for filtering.
     * @return The loaded, filtered and projjected table.
     */
    [[nodiscard]] std::unique_ptr<cudf::table> LoadTableSeq(const std::vector<std::string> &column_names,
                                                            std::function<int32_t()> chunk_num_gen,
                                                            const std::string &filter_column_name,
                                                            cudf::table_view filter_table,
                                                            rmm::cuda_stream_view stream);

    /**
     * Counts the number of rows in the given column without loading the data but just looking at the
     * metadata.
     * @param column_name The name of the column, typically a column with a primary key.
     * @return The number of rows in that column.
     */
    [[nodiscard]] std::size_t CountRows(const std::string &column_name);

    /**
     * Returns the number of chunks. This applies to all columns
     */
    [[nodiscard]] uint32_t num_chunks() const {
        if (columns_.empty()) {
            return 0;
        }
        return columns_.begin()->second.chunks.size();
    }

    /**
     * Returns the name of the table.
     */
    const std::string &table_name() const { return table_name_; }

    /**
     * Returns the name of the columns
     */
    const std::vector<std::string> column_names() const {
        std::vector<std::string> cols;

        for (auto &col : columns_) {
            cols.emplace_back(col.first);
        }
        // Ok to return a local object since the compiler will use copy-elision,
        // specifically named return value optimization, see:
        // https://en.cppreference.com/w/cpp/language/copy_elision
        return cols;
    }

    /**
     * Returns the column type.
     */
    [[nodiscard]] ColumnType column_type(std::string &column) const {
        if (columns_.empty()) {
            return ColumnType::Unspecified;
        }
        auto iter = columns_.find(column);
        if (iter == columns_.end()) {
            return ColumnType::Unspecified;
        }
        return iter->second.type;
    }

   private:
    using ChunkVector = std::vector<std::filesystem::path>;
    struct ColumnMeta {
        std::string column_name;
        ColumnType type;
        ChunkVector chunks;
    };

    /**
     * Initializes the table metadata.
     */
    void Init();

    /**
     * Populates the vector of file paths. Throws FileNotFound exception on error.
     */
    void EnumerateReadableFiles(std::vector<std::filesystem::path> &files);

    /**
     * Allocates memory on the device and loads the contents of the
     * file into that device buffer using Kvikio/GDS.
     * The resulting device buffer is returned or an exception is thrown.
     */
    std::unique_ptr<rmm::device_buffer> ReadFileKvikio(std::filesystem::path file,
                                                       size_t &read_bytes,
                                                       rmm::cuda_stream_view stream) const;

    std::string base_dir_;                       // The base directory that contains the dirs with
                                                 // the tables.
    std::string table_name_;                     // The name of the table.
    std::map<std::string, ColumnMeta> columns_;  // The columns, by column name.
};

}