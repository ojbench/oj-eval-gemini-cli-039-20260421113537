#include <iostream>
#include <vector>
#include <cassert>
#include "CSRMatrix.hpp"

int main() {
    using namespace sjtu;
    
    // Test basic constructor and get/set
    CSRMatrix<int> mat(3, 3);
    assert(mat.getRowSize() == 3);
    assert(mat.getColSize() == 3);
    assert(mat.getNonZeroCount() == 0);
    
    mat.set(0, 0, 1);
    mat.set(1, 1, 2);
    mat.set(2, 2, 3);
    assert(mat.getNonZeroCount() == 3);
    assert(mat.get(0, 0) == 1);
    assert(mat.get(1, 1) == 2);
    assert(mat.get(2, 2) == 3);
    assert(mat.get(0, 1) == 0);
    
    // Test dense constructor
    std::vector<std::vector<int>> dense = {
        {1, 0, 2},
        {0, 3, 0},
        {4, 0, 5}
    };
    CSRMatrix<int> mat2(3, 3, dense);
    assert(mat2.getNonZeroCount() == 5);
    assert(mat2.get(0, 0) == 1);
    assert(mat2.get(0, 2) == 2);
    assert(mat2.get(1, 1) == 3);
    assert(mat2.get(2, 0) == 4);
    assert(mat2.get(2, 2) == 5);
    
    // Test matrix-vector multiplication
    std::vector<int> vec = {1, 2, 3};
    std::vector<int> res = mat2 * vec;
    // [1 0 2] [1]   [1*1 + 0*2 + 2*3]   [7]
    // [0 3 0] [2] = [0*1 + 3*2 + 0*3] = [6]
    // [4 0 5] [3]   [4*1 + 0*2 + 5*3]   [19]
    assert(res[0] == 7);
    assert(res[1] == 6);
    assert(res[2] == 19);
    
    // Test getRowSlice
    CSRMatrix<int> slice = mat2.getRowSlice(0, 2);
    assert(slice.getRowSize() == 2);
    assert(slice.getColSize() == 3);
    assert(slice.getNonZeroCount() == 3);
    assert(slice.get(0, 0) == 1);
    assert(slice.get(0, 2) == 2);
    assert(slice.get(1, 1) == 3);
    
    // Test exceptions
    try {
        mat.get(3, 0);
        assert(false);
    } catch (const invalid_index &e) {}
    
    try {
        std::vector<int> vec2 = {1, 2};
        mat2 * vec2;
        assert(false);
    } catch (const size_mismatch &e) {}

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
