#include "Preprocessor/Preprocessor.h"
#include <iostream>

namespace shaderpulse {

namespace preprocessor {

void Preprocessor::process() {
    for (int i = 0; i < sourceCode.size(); i++) {
        char curChar = sourceCode[i];
        char nextChar = (i < (sourceCode.size()-1)) ? sourceCode[i+1] : 0;

        if (singleLineComment) {
            if (curChar == '\n') {
                singleLineComment = false;
                processedCode[i] = curChar;
            }

            continue;
        } else if (multiLineComment) {
            if (curChar == '\n') {
                processedCode[i] = curChar;
            } else if (curChar == '*' && nextChar == '/') {
                multiLineComment = false;
                i++;
            }

            continue;
        } else if (curChar == '/' && nextChar == '/') {
           singleLineComment = true;
        } else if (curChar == '/' && nextChar == '*') {
            multiLineComment = true;
        } else {
            processedCode[i] = sourceCode[i];
        }
    }
}

const std::string& Preprocessor::getProcessedSource() const {
    return processedCode;
}

}

}