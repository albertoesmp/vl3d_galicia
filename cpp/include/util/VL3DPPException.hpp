#pragma once

// ***   INCLUDES   *** //
// ******************** //
#include <exception>
#include <string>

namespace vl3dpp::util {


// ***   C L A S S   *** //
// ********************* //
/**
 * @author Alberto M. Esmoris Pena
 * @brief Class providing a baseline exception for the VL3D++ software.
 *
 * Baseline exception class for the VL3D++ software.
 */
class VL3DPPException : public std::exception {
protected:
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief The VL3DPPException's message.
     */
    std::string msg;

public:
    // ***   CONSTRUCTION/DESTRUCTION   *** //
    // ************************************ //
    /**
     * @brief Build a VL3DPPException with the given message that will be
     *  returned when calling the VL3DPPException::what method.
     * @param msg The message for the VL3DPPException.
     * @see VL3DPPException::what
     */
    VL3DPPException(std::string const &msg) : msg(msg) {}
    virtual ~VL3DPPException() = default;

    // ***  EXCEPTION METHODS  *** //
    // *************************** //
    /**
     * @brief Overwrites the std::exception::what method to return the
     *  message of the VL3DPPException.
     */
    virtual const char *what() const noexcept {
        return msg.c_str();
    }
};

}