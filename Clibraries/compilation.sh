#!/usr/bin/env bash

if gcc -c -fpic reader.c; then
    echo "C file compiled...";
else
    echo "File did not compile!";
    exit 0;
fi

if gcc -shared -o libreader.so reader.o; then
    echo "Shared library created...";
else
    echo "The shared library could not be created!";
    exit 0;
fi

if gcc -Wall -L/home/alves/Clibraries -lreader -L/home/alves/Clibraries/jsmn -ljsmn -o main main.c; then
    echo "Linking completed."
    ./main
else
    echo "The linking was not successfull!";
    exit 0;
fi 
