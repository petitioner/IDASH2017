################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/CipherGD.cpp \
../src/GD.cpp \
../src/IDASH2017.cpp \
../src/TestGD.cpp 

OBJS += \
./src/CipherGD.o \
./src/GD.o \
./src/IDASH2017.o \
./src/TestGD.o 

CPP_DEPS += \
./src/CipherGD.d \
./src/GD.d \
./src/IDASH2017.d \
./src/TestGD.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/home/john/Music/C16/IDASH2017-master/IDASH2017/HEAAN/HEAAN/src" -I"/home/john/Music/C16/IDASH2017-master/IDASH2017/lib/include" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


