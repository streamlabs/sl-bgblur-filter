## -- OBS Plugin

# Directory where this .cmake file lives
set(_this_dir "${CMAKE_CURRENT_LIST_DIR}")

include("${_this_dir}/cmake/FetchOnnxruntime.cmake")

add_library(sl-bgblur-filter MODULE)
add_library(OBS::sl-bgblur-filter ALIAS sl-bgblur-filter)

target_link_libraries(sl-bgblur-filter PRIVATE OBS::libobs)
target_link_libraries(sl-bgblur-filter PRIVATE Ort)
target_link_libraries(sl-bgblur-filter PRIVATE
     "${_this_dir}/opencv/opencv_imgproc481.lib"
     "${_this_dir}/opencv/opencv_core481.lib"
     "${_this_dir}/opencv/zlib.lib"

      # Debugging
     "${_this_dir}/opencv/opencv_imgcodecs480.lib"
     "${_this_dir}/opencv/libjpeg-turbo.lib"
     "${_this_dir}/opencv/libopenjp2.lib"
     "${_this_dir}/opencv/libpng.lib"
     "${_this_dir}/opencv/libtiff.lib"
     "${_this_dir}/opencv/IlmImf.lib"
)

target_include_directories(sl-bgblur-filter PRIVATE "${_this_dir}/opencv/include")

target_link_options(sl-bgblur-filter PRIVATE "/IGNORE:4099")

target_sources(sl-bgblur-filter PRIVATE
    "${_this_dir}/sl-bgblur-filter.cpp"
    "${_this_dir}/BgBlur.cpp"
    "${_this_dir}/OnnxModel.cpp"
    "${_this_dir}/OnnxInstance.cpp"
    "${_this_dir}/ModifyAppearance.cpp"
)

add_custom_command(TARGET sl-bgblur-filter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:Ort::DirectML>"
        "${_this_dir}/bgblurdata/SelfieMulticlass.onnx"
        "${_this_dir}/bgblurdata/mask_alpha_filter.effect"
        "${_this_dir}/bgblurdata/kawase_blur.effect"
        "${_this_dir}/bgblurdata/mask_color_correction.effect"
        $<TARGET_FILE_DIR:sl-bgblur-filter>
)

if(COMMAND set_target_properties_obs)
  set_target_properties_obs(sl-bgblur-filter PROPERTIES FOLDER plugins PREFIX "")
else()
  message(STATUS "set_target_properties_obs is not defined, skipping...")
endif()

install(FILES
    "${_this_dir}/bgblurdata/pphumanseg_fp32.onnx"
    "${_this_dir}/bgblurdata/mask_alpha_filter.effect"
    "${_this_dir}/bgblurdata/kawase_blur.effect"
    DESTINATION "${OBS_PLUGIN_DESTINATION}"
)
