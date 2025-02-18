#pragma once

#include <util/logging/BaseLogger.hpp>
#include <string>
#include <fstream>
#include <mutex>

using std::string;
using std::ofstream;

namespace vl3dpp::util::logging {

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief Basic logger which writes to standard out and specified file
 */
class BasicLogger : public BaseLogger{
protected:
    // ***  ATTRIBUTES  *** //
    // ******************** //
    /**
     * @brief Flag to specify if append to previously created log file (true)
     *  or not (false)
     */
    bool append = true;
    /**
     * @brief Flag to specify if output to file (true) or not (false)
     */
    bool outputToFile = true;
    /**
     * @brief Flag to specify if output to standard out (true) or not (false)
     */
    bool outputToStandardOut = true;
    /**
     * @brief Path to the output file
     */
    string path;
    /**
     * @brief File output stream
     */
    ofstream fos;
    /**
     * @brief Mutex to handle concurrent access to standard out
     */
    std::mutex stdoutMutex;
    /**
     * @brief Mutex to handle concurrent access to file output stream
     */
    std::mutex fosMutex;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Basic logger constructor
     * @see BasicLogger::path
     */
    BasicLogger(string path) : path(path) {}
    virtual ~BasicLogger() {}

    // ***  LOGGING FUNCTIONS  *** //
    // *************************** //
    /**
     * @see BaseLogger::outputLog
     */
    void outputLog(
        string const tag,
        string const timestamp,
        string const msg
    ) override;

    /**
     * @brief Output log message to standard out
     * @see BaseLogger::outputLog
     */
    virtual void logToStandardOut(
        string const tag,
        string const timestamp,
        string const msg
    );

    /**
     * @brief Output log message to file
     * @see BaseLogger::outputLog
     */
    virtual void logToFile(
        string const tag,
        string const timestamp,
        string const msg
    );

};

}
