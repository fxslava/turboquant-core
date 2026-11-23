include_guard(GLOBAL)
include(ExternalProject)

function(tq_scope_cann_wrapper_warnings)
  if(CMAKE_VERSION VERSION_LESS 3.19)
    message(FATAL_ERROR "CANN wrapper warning isolation requires CMake 3.19 or newer")
  endif()
  foreach(kernel IN LISTS ARGN)
    foreach(kind aic_device aiv_device device)
      set(wrapper "${kernel}_${kind}")
      if(TARGET "${wrapper}")
        ExternalProject_Add_Step("${wrapper}" tq_wrapper_warning_policy
            COMMAND "${CMAKE_COMMAND}" -S <SOURCE_DIR> -B <BINARY_DIR>
                "-DCMAKE_PROJECT_INCLUDE:FILEPATH=${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CannDeviceWrapperTargets.cmake"
            DEPENDEES configure
            DEPENDERS build
            ALWAYS TRUE)
      endif()
    endforeach()
  endforeach()
endfunction()
