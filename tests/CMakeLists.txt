add_library(facade_parser_test_support STATIC synthetic_facade.cpp)
target_include_directories(facade_parser_test_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(facade_parser_test_support PUBLIC facade_parser::facade_parser)

set(FACADE_PARSER_TEST_SOURCES
  test_edges.cpp
  test_periodicity.cpp
  test_grammar.cpp
  test_classification.cpp
  test_lattice_refine.cpp
  test_symmetry.cpp
  test_edges_export.cpp
  test_io_json.cpp
  test_end_to_end.cpp
)

foreach(test_src IN LISTS FACADE_PARSER_TEST_SOURCES)
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE facade_parser_test_support GTest::gtest_main)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
