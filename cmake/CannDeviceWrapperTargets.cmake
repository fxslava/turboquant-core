function(tq_apply_cann_wrapper_warning_policy)
  foreach(wrapper device_obj device_aic_obj device_aiv_obj)
    if(TARGET "${wrapper}")
      get_target_property(wrapper_sources "${wrapper}" SOURCES)
      foreach(source IN LISTS wrapper_sources)
        if(NOT source MATCHES "(^|/)auto_gen_[^/]+\\.cpp$")
          message(FATAL_ERROR "Refusing to modify non-wrapper source: ${source}")
        endif()
        # CANN 9.2.1's release wrapper keeps this argument even when overflow
        # writeback is compiled out. Consume it in generated code instead of
        # weakening -Werror or allowing a warning in a verified build.
        get_filename_component(wrapper_source "${source}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        file(READ "${wrapper_source}" wrapper_text)
        if(NOT wrapper_text MATCHES "TQ_WRAPPER_OVERFLOW_CONSUMED")
          string(REGEX REPLACE
              "(GM_ADDR[ \t]+overflow_status[ \t\r\n]*\\)[ \t\r\n]*\\{)"
              "\\1\n    (void)overflow_status; // TQ_WRAPPER_OVERFLOW_CONSUMED"
              consumed_wrapper_text "${wrapper_text}")
          if(NOT consumed_wrapper_text STREQUAL wrapper_text)
            file(WRITE "${wrapper_source}" "${consumed_wrapper_text}")
          endif()
        endif()
      endforeach()
    endif()
  endforeach()
endfunction()

cmake_language(DEFER CALL tq_apply_cann_wrapper_warning_policy)
