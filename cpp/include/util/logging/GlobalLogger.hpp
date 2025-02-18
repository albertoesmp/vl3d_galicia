#pragma once

#include <util/logging/BaseLogger.hpp>
#include <util/logging/BasicLogger.hpp>

#include <memory>

using std::shared_ptr;
using std::make_shared;
using vl3dpp::util::logging::BasicLogger;

extern shared_ptr<BasicLogger> LOGGER;
