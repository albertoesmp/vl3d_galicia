#include <test/BaseTest.hpp>
#include <fstream>

// ***   STATIC UTILS   *** //
// ************************ //
template <typename NumericType>
std::vector<arma::Col<NumericType>> BaseTest::readRaggedMatrix(
    std::string const &path,
    std::string const sep
){
    // Prepare output data structure
    std::vector<arma::Col<NumericType>> out;
    std::vector<NumericType> buffer;
    // Read file to populate data structure
    std::ifstream ifs(path, std::ios::in);
    std::string line;
    while(std::getline(ifs, line)){ // For each line
        size_t startIdx = 0;
        size_t const endIdx = line.size();
        buffer.clear();
        while(startIdx < endIdx){
            size_t const sepIdx = line.find(sep, startIdx);
            buffer.push_back(
                (NumericType) std::stod(line.substr(startIdx, sepIdx).c_str())
            );
            // Update start index for next iteration, if any
            startIdx = (sepIdx==std::string::npos) ? endIdx : sepIdx+sep.size();
        }
        out.push_back(arma::conv_to<arma::Col<NumericType>>::from(buffer));
    }
    // Return
    return out;
}
