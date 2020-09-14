# Libtorch Image Recognition
============

This project is training neural network model to recognize images, TV in this case, among other objects inside my flat. I used Facebook libtorch to make it happen.

---

## Features
- Custom libtorch image dataset

---


## Setup
I installed libtorch in a workspace so that you do not need to do extra work.
ADDITION: Unfortunatelly I couldn't submit project with libtorch preinstalled. Please, download libtorch from the https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip and put it in /home/workspace folder.
You can build the project by issuing the following commands from root directory of the project:
```
cd build/
cmake ..
make
```
If you compiling from local machine you need to install OpenCV and libtorch. And put these lines in CmakeLists.txt:
```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ir)

set(ENV{Torch_DIR} <path to libtorch>)
set(ENV{OPENCV_DIR} <path to OpenCV>)

find_package( OpenCV REQUIRED )

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)


file(GLOB project_SRCS src/*.cpp)

add_executable(ir ${project_SRCS})

target_link_libraries(ir "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET ir PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)

```

---

## Structure

### File structure
dataset.cpp - implements custom dataset.
csv.cpp - implements csv reader.
model.cpp - libtorch neural network model implemented as struct.
recognition.cpp - main execution point with main(). The core logic of an application.

### Class structure
Although struct Net in model.cpp, this struct clearly exposes main Object Oriented Programming concepts.

---

## Rubric

csv.cpp - implements csv reader.
Line 18-20. The project reads data from an external file or writes data to a file as part of the necessary operation of the program.

model.cpp:
Line 3. The project code is organized into classes with class attributes to hold the data, and class methods to perform tasks. Considering that struct is some cheap form of class. 
Line 4-7. All class members that are set to argument values are initialized through member initialization lists.

recognition.cpp:
Line 20. The project code is clearly organized into functions.
Line 89-92. Line 32-49. A variety of control structures are used in the project.


---

## Sources

I used https://github.com/mhubii/libtorch_custom_dataset to implement custom dataset.

---

## Usage

In order to train a model please put images in /data folder and path to these files in train.csv along with labels. train.csv format <name of the file>, <label>.
Then start an application:
```
./ir
```
Expected output:
```
Training on CPU.
Train Epoch: 1 [6/6] Loss: 77386687184896.0000
Epoch ended
Train Epoch: 2 [6/6] Loss: 0.6853433699952787456.00000000
Epoch ended
```

---
