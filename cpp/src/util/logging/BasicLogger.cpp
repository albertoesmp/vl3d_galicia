#include <util/logging/BasicLogger.hpp>

#include <sstream>
#include <iostream>

using namespace vl3dpp::util::logging;
using std::cout;
// ***  LOGGING FUNCTIONS  *** //
// *************************** //
void BasicLogger::outputLog(
    string const tag,
    string const timestamp,
    string const msg
){
    if(outputToStandardOut) logToStandardOut(tag, timestamp, msg);
    if(outputToFile) logToFile(tag, timestamp, msg);
}

void BasicLogger::logToStandardOut(
    string const tag,
    string const timestamp,
    string const msg
){
    std::stringstream ss;
    ss << timestamp << " (" << tag << ")" << ": " << msg << std::endl;
    std::unique_lock<std::mutex> unilock(stdoutMutex);
    cout << ss.str();
}

void BasicLogger::logToFile(
    string const tag,
    string const timestamp,
    string const msg
){
    std::stringstream ss;
    ss << timestamp << " (" << tag << ")" << ": " << msg << std::endl;
    std::unique_lock<std::mutex> unilock(fosMutex);
    if(!fos.is_open()){
        if(append) fos.open(path, std::ios_base::app);
        else fos.open(path, std::ios_base::out);
    }
    fos << ss.str();
    fos.flush();
}
