#pragma once
#include <string>

namespace shaderpulse {

namespace preprocessor {

class Preprocessor {
public:
  Preprocessor(const std::string &sourceCode)
      : sourceCode(sourceCode), 
        processedCode(std::string(sourceCode.length(), ' ')),
        singleLineComment(false),
        multiLineComment(false) {

      }

    void process();
    const std::string& getProcessedSource() const;

private:
    std::string sourceCode;
    std::string processedCode;
    bool singleLineComment;
    bool multiLineComment;
};

} // namespace shaderpulse

} // namespace preprocessor