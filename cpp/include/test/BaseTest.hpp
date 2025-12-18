#ifndef BASE_TEST_HPP_
#define BASE_TEST_HPP_

#include <armadillo>
#include <vector>
#include <string>
#include <ostream>

namespace vl3dpp { namespace test {

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief BaseTest class
 *
 * Can be overridden to implement new tests.
 *
 * NOTE that any test that must be directly runnable must override the run
 * method with its custom testing logic.
 */
class BaseTest {
protected:
    // ***  ATTRIBUTES  *** //
    // ******************** //
    /**
     * @brief The name of the test.
     */
    std::string const name;
    /**
     * @brief Flag representing whether the test has been correctly
     *  constructed (true) or not (false).
     */
    bool correctlyConstructed;

public:
    // ***  CONSTRUCTION / DESTRUCTION   *** //
    // ************************************* //
    /**
     * @brief Base test constructor.
     * @param name The name of the test.
     * @see BaseTest::name
     */
    BaseTest(std::string const &name) :
        name(name),
        correctlyConstructed(true)
    {}
    virtual ~BaseTest() = default;

    // ***  GETTERs and SETTERs  *** //
    // ***************************** //
    /**
     * @brief Obtain the test's name.
     * @return The test's name.
     * @see BaseTest::name
     */
    inline std::string getName() const {return name;}

    // ***  R U N  *** //
    // *************** //
    /**
     * @brief The test's logic.
     *
     * @return True if test was successfully passed, false otherwise.
     */
    virtual bool run() = 0;

    /**
     * @brief Do the test and report its status (whether passed or not)
     *  through the given output stream.
     * @param out The output stream to report the test status.
     * @param color Whether to include colors in the output stream (true) or
     *  not (false).
     * @return True if test was successfully passed, false otherwise.
     * @see BaseTest::run
     */
    bool operator() (std::ostream &out, bool const color);

    // ***   STATIC UTILS   *** //
    // ************************ //
    /**
     * @brief Read a numeric CSV where each row can have a different number of
     *  columns.
     * @tparam NumericType Data type for the numbers.
     * @param path Path to the CSV file to be read.
     * @param sep The separator for the CSV (default ",").
     * @return Read CSV as a ragged matrix where the standard vector
     *  has the rows as elements and the internal armadillo vector represents
     *  the columns.
     */
    template <typename NumericType>
    static std::vector<arma::Col<NumericType>> readRaggedMatrix(
        std::string const &path,
        std::string const sep=","
    );
};

#include <test/BaseTest.tpp>

}}

#endif