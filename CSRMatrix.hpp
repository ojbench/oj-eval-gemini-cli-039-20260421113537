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

// TODO: Implement a CSR matrix class
// You only need to implement the TODOs in this file
// DO NOT modify other parts of this file
// DO NOT include any additional headers
// DO NOT use STL other than std::vector

template <typename T>
class CSRMatrix {

private:
    size_t rows;
    size_t cols;
    std::vector<size_t> indptr;
    std::vector<size_t> indices;
    std::vector<T> data;
    
public:
    // Assignment operators are deleted
    CSRMatrix &operator=(const CSRMatrix &other) = delete;
    CSRMatrix &operator=(CSRMatrix &&other) = delete;

    // Constructor for empty matrix with dimensions
    // TODO: Initialize an empty CSR matrix with n rows and m columns
    CSRMatrix(size_t n, size_t m) : rows(n), cols(m), indptr(n + 1, 0) {}

    // Constructor with pre-built CSR components
    // TODO: Initialize CSR matrix from existing CSR format data, validate sizes
    CSRMatrix(size_t n, size_t m, size_t count,
        const std::vector<size_t> &indptr, 
        const std::vector<size_t> &indices,
        const std::vector<T> &data) 
        : rows(n), cols(m), indptr(indptr), indices(indices), data(data) {
        if (this->indptr.size() != n + 1) throw size_mismatch();
        if (this->indices.size() != count || this->data.size() != count) throw size_mismatch();
        if (this->indptr[0] != 0 || this->indptr[n] != count) throw size_mismatch();
    }

    // Copy constructor
    CSRMatrix(const CSRMatrix &other) = default;

    // Move constructor
    CSRMatrix(CSRMatrix &&other) = default;

    // Constructor from dense matrix format (given as vector of vectors)
    // TODO: Convert dense matrix representation to CSR format
    CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &dense_data) 
        : rows(n), cols(m) {
        if (dense_data.size() != n) throw size_mismatch();
        indptr.reserve(n + 1);
        indptr.push_back(0);
        for (size_t i = 0; i < n; ++i) {
            if (dense_data[i].size() != m) throw size_mismatch();
            for (size_t j = 0; j < m; ++j) {
                if (dense_data[i][j] != T()) {
                    indices.push_back(j);
                    data.push_back(dense_data[i][j]);
                }
            }
            indptr.push_back(indices.size());
        }
    }

    // Destructor
    ~CSRMatrix() = default;

    // Get dimensions and non-zero count
    // TODO: Return the number of rows
    size_t getRowSize() const { return rows; }

    // TODO: Return the number of columns
    size_t getColSize() const { return cols; }

    // TODO: Return the count of non-zero elements
    size_t getNonZeroCount() const { return data.size(); }

    // Element access
    // TODO: Retrieve element at position (i,j)
    T get(size_t i, size_t j) const {
        if (i >= rows || j >= cols) throw invalid_index();
        size_t start = indptr[i];
        size_t end = indptr[i+1];
        for (size_t k = start; k < end; ++k) {
            if (indices[k] == j) return data[k];
            if (indices[k] > j) break;
        }
        return T();
    }

    // TODO: Set element at position (i,j), updating CSR structure as needed
    void set(size_t i, size_t j, const T &value) {
        if (i >= rows || j >= cols) throw invalid_index();
        
        size_t start = indptr[i];
        size_t end = indptr[i+1];
        
        size_t pos = start;
        bool found = false;
        while (pos < end) {
            if (indices[pos] == j) {
                found = true;
                break;
            }
            if (indices[pos] > j) {
                break;
            }
            pos++;
        }
        
        if (found) {
            data[pos] = value;
        } else {
            indices.insert(indices.begin() + pos, j);
            data.insert(data.begin() + pos, value);
            for (size_t r = i + 1; r <= rows; ++r) {
                indptr[r]++;
            }
        }
    }

    // Access CSR components
    // TODO: Return the row pointer array
    const std::vector<size_t> &getIndptr() const { return indptr; }

    // TODO: Return the column indices array
    const std::vector<size_t> &getIndices() const { return indices; }

    // TODO: Return the data values array
    const std::vector<T> &getData() const { return data; }

    // Convert to dense matrix format
    // TODO: Convert CSR format to dense matrix representation
    std::vector<std::vector<T>> getMatrix() const {
        std::vector<std::vector<T>> res(rows, std::vector<T>(cols, T()));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = indptr[i]; k < indptr[i+1]; ++k) {
                res[i][indices[k]] = data[k];
            }
        }
        return res;
    }

    // Matrix-vector multiplication
    // TODO: Implement multiplication of this matrix with vector vec
    std::vector<T> operator*(const std::vector<T> &vec) const {
        if (vec.size() != cols) throw size_mismatch();
        std::vector<T> res(rows, T());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = indptr[i]; k < indptr[i+1]; ++k) {
                res[i] += data[k] * vec[indices[k]];
            }
        }
        return res;
    }

    // Row slicing
    // TODO: Extract submatrix containing rows [l,r)
    CSRMatrix getRowSlice(size_t l, size_t r) const {
        if (l > r || r > rows) throw invalid_index();
        size_t n = r - l;
        std::vector<size_t> new_indptr;
        new_indptr.reserve(n + 1);
        size_t offset = indptr[l];
        for (size_t i = l; i <= r; ++i) {
            new_indptr.push_back(indptr[i] - offset);
        }
        
        size_t start_idx = indptr[l];
        size_t end_idx = indptr[r];
        size_t count = end_idx - start_idx;
        
        std::vector<size_t> new_indices;
        new_indices.reserve(count);
        for (size_t i = start_idx; i < end_idx; ++i) {
            new_indices.push_back(indices[i]);
        }
        
        std::vector<T> new_data;
        new_data.reserve(count);
        for (size_t i = start_idx; i < end_idx; ++i) {
            new_data.push_back(data[i]);
        }
        
        return CSRMatrix(n, cols, count, new_indptr, new_indices, new_data);
    }
};

}

#endif // CSR_MATRIX_HPP

