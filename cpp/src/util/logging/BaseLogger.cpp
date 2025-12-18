#include <util/logging/BaseLogger.hpp>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

using namespace std::chrono;
using std::time_t;
using std::stringstream;
using std::fixed;
using std::setfill;
using std::setw;

using namespace vl3dpp::util::logging;

// ***  STATIC CONSTANTS  *** //
// ************************** //
string const BaseLogger::LOGTAG_ERRO = "ERRO";
string const BaseLogger::LOGTAG_INFO = "INFO";
string const BaseLogger::LOGTAG_WARN = "WARN";
string const BaseLogger::LOGTAG_DEBG = "DEBG";
string const BaseLogger::LOGTAG_XTRA = "XTRA";

// ***  LOGGING FUNCTIONS  *** //
// *************************** //
void BaseLogger::logError(string const msg){
    if(enableError) outputLog(LOGTAG_ERRO, buildTimestamp(), msg);
};
void BaseLogger::logInfo(string const msg){
    if(enableInfo) outputLog(LOGTAG_INFO, buildTimestamp(), msg);
}
void BaseLogger::logWarn(string const msg){
    if(enableWarn) outputLog(LOGTAG_WARN, buildTimestamp(), msg);
}
void BaseLogger::logDebug(string const msg){
    if(enableDebug) outputLog(LOGTAG_DEBG, buildTimestamp(), msg);
}
void BaseLogger::logExtra(string const msg){
    if(enableExtra) outputLog(LOGTAG_XTRA, buildTimestamp(), msg);
}

// ***  UTIL FUNCTIONS  *** //
// ************************ //
string BaseLogger::buildTimestamp(){
    if(includeTimestamp) return _buildTimestamp();
    return "";
}
string BaseLogger::_buildTimestamp(){
    system_clock::time_point tp = system_clock::now();
    time_t t = system_clock::to_time_t(tp);
    struct std::tm *ts = localtime(&t);
    stringstream ss;
    ss  << "["
        << setw(4) << fixed << setfill('0') << 1900+ts->tm_year << "-"
        << setw(2) << fixed << setfill('0') << ts->tm_mon + 1 << "-"
        << setw(2) << fixed << setfill('0') << ts->tm_mday << " "
        << setw(2) << fixed << setfill('0') << ts->tm_hour << ":"
        << setw(2) << fixed << setfill('0') << ts->tm_min << ":"
        << setw(2) << fixed << setfill('0') << ts->tm_sec
        << "]";
    return ss.str();

}

// ***  CONFIGURATION FUNCTIONS  *** //
// ********************************* //
void BaseLogger::silence(){
    enableError = false;
    enableInfo = false;
    enableWarn = false;
    enableDebug = false;
    enableExtra = false;
}

void BaseLogger::fullLogging(){
    enableError = true;
    enableInfo = true;
    enableWarn = true;
    enableDebug = true;
    enableExtra= true;
}
