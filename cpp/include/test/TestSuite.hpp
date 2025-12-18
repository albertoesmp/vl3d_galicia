#pragma once

#include <test/BaseTest.hpp>
#include <vector>
#include <memory>
#include <functional>

namespace vl3dpp { namespace test{

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief TestSuite class
 *
 * Provides a way to group related tests together into a suite. All the tests
 * will be ran one-by-one, i.e., in a sequential way. If all tests in the
 * suite are passed, the suite is passed. If at least one test in the suite
 * fails, then the suite is considered as failed.
 */
class TestSuite{
protected:
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief The name of the suite.
     */
    std::string name;
    /**
     * @brief The tests composing the suite.
     */
    std::vector<std::shared_ptr<BaseTest>> tests;
    /**
     * @brief How many tests failed.
     *
     * It will be updated with the number of failed tests when TestSuite::run
     * is called.
     */
    int failedTests;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Base test suite constructor.
     * @param name The name of the suite.
     * @see TestSuite::name
     */
    TestSuite(std::string const &name) : name(name) {}

    // ***  GETTERs and SETTERs  *** //
    // ***************************** //
    /**
     * @brief Obtain the suite's name.
     * @return The suite's name.
     * @see TestSuite::name
     */
    inline std::string getName() const {return name;}
    /**
     * @brief Add the given test to the suite.
     * @param test Test to be added to the suite.
     * @see TestSuite::tests
     */
    inline void addTest(std::shared_ptr<BaseTest> test)
    {tests.push_back(test);}

    /**
     * @brief Get the number of failed tests.
     * @return Number of failed tests.
     * @see TestSuite::failedTests
     */
    inline int getFailedCount() const
    {return failedTests;}

    // ***   R U N   *** //
    // ***************** //
    /**
     * @brief Run the suite.
     * @return True if all the tests in the suite are passed, false otherwise.
     * @see BaseTest::run
     */
    bool run();

    /**
     * @brief Do the test suite and report its final status (whether passed or
     *  not) through the given output stream.
     * @param out The output stream to report the suite status.
     * @param color Whether to include colors in the output stream (true) or
     *  not (false).
     * @return True if suite was successfully passed, false otherwise.
     * @see TestSuite::run
     * @see BaseTest::operator()
     */
    bool operator() (std::ostream &out, bool const color);

protected:
    /**
     * @brief Common logic for TestSuite::run and TestSuite::operator()
     * @return True if suite was successfully passed, false otherwise.
     * @see TestSuite::run
     * @see TestSuite::operator()
     */
    bool _run(std::function<bool(std::shared_ptr<BaseTest>)> testCaller);
};

}}