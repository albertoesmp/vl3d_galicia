#pragma once

#include <string>

using std::string;

namespace vl3dpp::util::logging {

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief Base logger
 */
class BaseLogger {
public:
    // ***  STATIC CONSTANTS  *** //
    // ************************** //
    /**
     * @brief Tag for log error messages
     */
    static string const LOGTAG_ERRO;
    /**
     * @brief Tag for log information messages
     */
    static string const LOGTAG_INFO;
    /**
     * @brief Tag for log warning messages
     */
    static string const LOGTAG_WARN;
    /**
     * @brief Tag for log debug messages
     */
    static string const LOGTAG_DEBG;
    /**
     * @brief Tag for log extra messages
     */
    static string const LOGTAG_XTRA;

protected:
    // ***  ATTRIBUTES  *** //
    // ******************** //
    /**
     * @brief Flag to specify if include timestamp (true) or not (false)
     */
    bool includeTimestamp = true;

public:
    /**
     * @brief Flag to specify if ignore error logging (false) or not (true)
     */
    bool enableError = true;
    /**
     * @brief Flag to specify if ignore info logging (false) or not (true)
     */
    bool enableInfo = true;
    /**
     * @brief Flag to specify if ignore warn logging (false) or not (true)
     */
    bool enableWarn = true;
    /**
     * @brief Flag to specify if ignore debug logging (false) or not (true)
     */
    bool enableDebug = true;
    /**
     * @brief Flag to specify if ignore extra logging (false) or not (true)
     */
    bool enableExtra = true;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Base logger constructor
     */
    BaseLogger() = default;
    virtual ~BaseLogger() {}

    // ***  LOGGING FUNCTIONS  *** //
    // *************************** //
    /**
     * @brief Log error message
     * @param msg Error message
     */
    void logError(string const msg);
    /**
     * @brief Log information message
     * @param msg Information message
     */
    void logInfo(string const msg);
    /**
     * @brief Log warning message
     * @param msg Warning message
     */
    void logWarn(string const msg);
    /**
     * @brief Log debug message
     * @param msg Debug message
     */
    void logDebug(string const msg);
    /**
     * @brief Log extra message
     * @param msg Extra message
     */
    void logExtra(string const msg);

protected:
    /**
     * @brief Output log message considering given tag and timestamp
     *
     * This function must be overridden by all concrete logger implementations,
     *  since it is assuming responsibility of making log output effective
     *
     * @param tag Tag for the log message
     * @param timestamp Timestamp for the log message. It can be an empty
     *  string "", so no timestamp will be outputed
     * @param msg Log message itself
     */
    virtual void outputLog(
        string const tag,
        string const timestamp,
        string const msg
    ) = 0;

    // ***  UTIL FUNCTIONS  *** //
    // ************************ //
    /**
     * @brief Build timestamp string with current time if include timestamp
     *  is setted to true, otherwise nothing will be built
     * @return Timestamp string with current time or empty string ""
     */
    string buildTimestamp();
    /**
     * @brief Build timestamp string with current time
     * @return Timestamp string with current time
     */
    string _buildTimestamp();

public:
    // ***  CONFIGURATION FUNCTIONS  *** //
    // ********************************* //
    /**
     * @brief Disable all logging modes so nothing will be outputted
     */
    void silence();
    /**
     * @brief Enable al logging modes so everything will be outputted
     */
    void fullLogging();
};

}