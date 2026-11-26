# Device CTests are opt-in and serialized on device 0. Build targets never run them.
function(tq_register_device_test name target)
  if(NOT TQ_REGISTER_DEVICE_TESTS)
    return()
  endif()
  cmake_parse_arguments(TQ "" "" "ARGS;LABELS" ${ARGN})
  add_test(NAME ${name} COMMAND $<TARGET_FILE:${target}> ${TQ_ARGS})
  set(artifacts "${CMAKE_BINARY_DIR}/device-test-artifacts/${name}")
  file(MAKE_DIRECTORY "${artifacts}")
  set_tests_properties(${name} PROPERTIES
      LABELS "device;${RUN_MODE};${TQ_LABELS}"
      RUN_SERIAL TRUE WORKING_DIRECTORY "${artifacts}")
  if(RUN_MODE STREQUAL "sim")
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "${TQ_SIM_TEST_ENV}")
  endif()
endfunction()
