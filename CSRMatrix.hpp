#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>
#include <exception>

namespace sjtu {

class size_mismatch : public std::exception {
public:
    const char *what() const noexcept override {
        return "Size mismatch";
    }
};

class invalid_index : public std::exception {
public:
    const char *what() const noexcept override {
        return "Index out of range";
    }
};

template <typename T>
class CSRMatrix {

private:
    size_t rows;
    size_t cols;
    // Dual-storage approach:
    // We store data in per-row vectors (row_indices and row_data) to allow
    // efficient element insertion (set operation) and row-based operations.
    // Flat CSR components (flat_indptr, flat_indices, flat_data) are maintained
    // lazily and rebuilt only when requested via getIndptr(), getIndices(), or getData().
    std::vector<std::vector<size_t>> row_indices;
    std::vector<std::vector<T>> row_data;
    
    mutable std::vector<size_t> flat_indptr;
    mutable std::vector<size_t> flat_indices;
    mutable std::vector<T> flat_data;
    mutable bool dirty;

    // Rebuilds the flat CSR components from per-row storage if the data has changed.
    void flatten() const {
        if (!dirty) return;
        flat_indptr.assign(rows + 1, 0);
        size_t total_nnz = 0;
        for (size_t i = 0; i < rows; ++i) {
            total_nnz += row_indices[i].size();
        }
        flat_indices.clear();
        flat_data.clear();
        flat_indices.reserve(total_nnz);
        flat_data.reserve(total_nnz);
        for (size_t i = 0; i < rows; ++i) {
            flat_indptr[i] = flat_indices.size();
            for (size_t k = 0; k < row_indices[i].size(); ++k) {
                flat_indices.push_back(row_indices[i][k]);
                flat_data.push_back(row_data[i][k]);
            }
        }
        flat_indptr[rows] = flat_indices.size();
        dirty = false;
    }
    
public:
    CSRMatrix &operator=(const CSRMatrix &other) = delete;
    CSRMatrix &operator=(CSRMatrix &&other) = delete;

    CSRMatrix(size_t n, size_t m) : rows(n), cols(m), row_indices(n), row_data(n), dirty(true) {}

    CSRMatrix(size_t n, size_t m, size_t count,
        const std::vector<size_t> &indptr, 
        const std::vector<size_t> &indices,
        const std::vector<T> &data) 
        : rows(n), cols(m), row_indices(n), row_data(n), dirty(true) {
        if (indptr.size() != n + 1) throw size_mismatch();
        if (indices.size() != count || data.size() != count) throw size_mismatch();
        if (indptr[0] != 0 || indptr[n] != count) throw size_mismatch();
        
        for (size_t i = 0; i < n; ++i) {
            size_t start = indptr[i];
            size_t end = indptr[i+1];
            row_indices[i].reserve(end - start);
            row_data[i].reserve(end - start);
            for (size_t k = start; k < end; ++k) {
                row_indices[i].push_back(indices[k]);
                row_data[i].push_back(data[k]);
            }
        }
    }

    CSRMatrix(const CSRMatrix &other) 
        : rows(other.rows), cols(other.cols), 
          row_indices(other.row_indices), row_data(other.row_data), 
          dirty(true) {}

    CSRMatrix(CSRMatrix &&other) 
        : rows(other.rows), cols(other.cols), row_indices(other.rows), row_data(other.rows), dirty(true) {
        row_indices.swap(other.row_indices);
        row_data.swap(other.row_data);
    }

    CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &dense_data) 
        : rows(n), cols(m), row_indices(n), row_data(n), dirty(true) {
        if (dense_data.size() != n) throw size_mismatch();
        for (size_t i = 0; i < n; ++i) {
            if (dense_data[i].size() != m) throw size_mismatch();
            for (size_t j = 0; j < m; ++j) {
                if (dense_data[i][j] != T()) {
                    row_indices[i].push_back(j);
                    row_data[i].push_back(dense_data[i][j]);
                }
            }
        }
    }

    ~CSRMatrix() = default;

    size_t getRowSize() const { return rows; }
    size_t getColSize() const { return cols; }
    size_t getNonZeroCount() const {
        size_t count = 0;
        for (size_t i = 0; i < rows; ++i) {
            count += row_indices[i].size();
        }
        return count;
    }

    // Retrieve element at position (i,j) using binary search on the row's indices.
    T get(size_t i, size_t j) const {
        if (i >= rows || j >= cols) throw invalid_index();
        const auto &r_indices = row_indices[i];
        const auto &r_data = row_data[i];
        size_t low = 0;
        size_t high = r_indices.size();
        while (low < high) {
            size_t mid = low + (high - low) / 2;
            if (r_indices[mid] == j) return r_data[mid];
            if (r_indices[mid] < j) low = mid + 1;
            else high = mid;
        }
        return T();
    }

    // Set element at position (i,j). Uses binary search to find the position.
    // If the element exists, it is updated. Otherwise, it is inserted.
    void set(size_t i, size_t j, const T &value) {
        if (i >= rows || j >= cols) throw invalid_index();
        auto &r_indices = row_indices[i];
        auto &r_data = row_data[i];
        size_t low = 0;
        size_t high = r_indices.size();
        while (low < high) {
            size_t mid = low + (high - low) / 2;
            if (r_indices[mid] == j) {
                r_data[mid] = value;
                dirty = true;
                return;
            }
            if (r_indices[mid] < j) low = mid + 1;
            else high = mid;
        }
        r_indices.insert(r_indices.begin() + low, j);
        r_data.insert(r_data.begin() + low, value);
        dirty = true;
    }

    const std::vector<size_t> &getIndptr() const {
        flatten();
        return flat_indptr;
    }

    const std::vector<size_t> &getIndices() const {
        flatten();
        return flat_indices;
    }

    const std::vector<T> &getData() const {
        flatten();
        return flat_data;
    }

    std::vector<std::vector<T>> getMatrix() const {
        std::vector<std::vector<T>> res(rows, std::vector<T>(cols, T()));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < row_indices[i].size(); ++k) {
                res[i][row_indices[i][k]] = row_data[i][k];
            }
        }
        return res;
    }

    // Matrix-vector multiplication.
    // O(NNZ) complexity, where NNZ is the number of non-zero elements.
    std::vector<T> operator*(const std::vector<T> &vec) const {
        if (vec.size() != cols) throw size_mismatch();
        std::vector<T> res(rows, T());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < row_indices[i].size(); ++k) {
                res[i] += row_data[i][k] * vec[row_indices[i][k]];
            }
        }
        return res;
    }

    // Extract submatrix containing rows [l,r).
    // Efficiently implemented by copying only the relevant row vectors.
    CSRMatrix getRowSlice(size_t l, size_t r) const {
        if (l > r || r > rows) throw invalid_index();
        size_t n = r - l;
        CSRMatrix res(n, cols);
        for (size_t i = 0; i < n; ++i) {
            res.row_indices[i] = row_indices[l + i];
            res.row_data[i] = row_data[l + i];
        }
        return res;
    }
};

}

#endif // CSR_MATRIX_HPP