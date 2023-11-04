cmake_minimum_required(VERSION 3.12)
project(LShapeAnalyser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(MKL_LIBRARIES "C:/")
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED COMPONENTS core highgui)
find_package(PCL REQUIRED)
find_package(CGAL REQUIRED)
find_package(Torch REQUIRED)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(SOURCES
    src/main.cpp
    src/LShapeAnalyser.cpp
    src/LShapeAnalyser_single.cpp
    src/LShapeAnalyser_multiple.cpp
    src/LShapeAnalyserFunction.cpp
    src/OrbifoldTutteEmbedding.cpp
    src/PosConstraints.cpp
)

set(HEADERS
    include/LShapeAnalyser.h
    include/LShapeAnalyser_single.h
    include/LShapeAnalyser_multiple.h
    include/LShapeAnalyserFunction.h
    include/OrbifoldTutteEmbedding.h
    include/PosConstraints.h
)

set(UI
    ui/LShapeAnalyser.ui
    ui/LShapeAnalyser_single.ui
    ui/LShapeAnalyser_multiple.ui
)

qt5_wrap_cpp(MOC_SOURCES ${HEADERS})
qt5_wrap_ui(UI_HEADERS ${UI})

add_executable(${PROJECT_NAME} ${SOURCES} ${MOC_SOURCES} ${UI_HEADERS})
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    Qt5::Widgets
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    CGAL::CGAL
    ${TORCH_LIBRARIES}
)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND windeployqt $<TARGET_FILE:${PROJECT_NAME}>
)

message(STATUS ${CMAKE_LIBRARY_PATH})