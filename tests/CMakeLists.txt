set(BOOST_COMPONENTS unit_test_framework)
search_for_boost()

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories(${Boost_INCLUDE_DIRS})

add_executable(QPTest QPTest.cpp)
target_link_libraries(QPTest ${Boost_LIBRARIES} ${PROJECT_NAME})
add_test(QPTestUnit QPTest)
