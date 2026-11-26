# Sealed packet reader and dequantizer. No device encoder is included.
tq_add_ascend_kernel(TARGET tq_e8_reader RUN_MODE ${RUN_MODE}
    SOURCES "${PROJECT_SOURCE_DIR}/src/host/asc_e8_reader.cpp"
    DEFS ENABLE_CV_COMM_VIA_SSBUF=1
    DEVICE_OPTIONS -fno-fast-math -ffp-contract=off
    HOST_OPTIONS -fno-fast-math -ffp-contract=off)
tq_add_ascend_kernel(TARGET tq_e8_dequant RUN_MODE ${RUN_MODE}
    SOURCES "${PROJECT_SOURCE_DIR}/src/host/asc_e8_component.cpp"
    DEFS ENABLE_CV_COMM_VIA_SSBUF=1
    DEVICE_OPTIONS -fno-fast-math -ffp-contract=off
    HOST_OPTIONS -fno-fast-math -ffp-contract=off)
add_custom_target(tq_e8_kernels DEPENDS tq_e8_reader tq_e8_dequant)
