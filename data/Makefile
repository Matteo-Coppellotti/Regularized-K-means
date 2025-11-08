# Makefile

# Configuration
CXX = g++
CXXFLAGS = -O2 -g -std=c++20 -march=native
INCLUDES = -I/home/matteo/Desktop/eigen -I../fdaPDE-cpp -I../fdaPDE-cpp/fdaPDE/core
SRC = ./src/gen_data.cpp #################################################################
BUILD_DIR = build
TARGET = $(BUILD_DIR)/main
EXECUTABLE = main

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SRC)

# Run the program
run: all
	@cd $(BUILD_DIR) && \
	./$(EXECUTABLE)

# Clean up build files
clean:
	@rm -rf $(BUILD_DIR)

.PHONY: all run clean