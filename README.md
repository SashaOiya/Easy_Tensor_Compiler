# Easy Tensor Compiler
Для работы с моей программой вам понадобятся
- CMake of version 3.20 (or higher)
- python3
- python3-venv
- pip

# Установка
Склонируйте репозиторий, перейдите в папку с ним.
Сборка проекта:

## Установите необходимые зависимости:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
## Соберите проект
```sh
conan install . --output-folder=third_party --build=missing -s compiler.cppstd=23
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=./third_party/conan_toolchain.cmake
cd build/
make
```

## Benchmark :
Чтобы запустить бенчмарки, перейдите в папку "build" и напишите :
```sh
./benchmark/matmul_benchmark
```

