################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../imc/MultilayerPerceptron.cpp \
../imc/util.cpp 

OBJS += \
./imc/MultilayerPerceptron.o \
./imc/util.o 

CPP_DEPS += \
./imc/MultilayerPerceptron.d \
./imc/util.d 


# Each subdirectory must supply rules for building sources it contributes
imc/%.o: ../imc/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/home/javier/eclipse-workspace/la2" -include"/home/javier/eclipse-workspace/la2" -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


