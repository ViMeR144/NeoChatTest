#include <iostream>
#include <string>
#include <map>
#include <algorithm>

int main(int argc, char* argv[]) {
    std::cout << "DEBUG: Program started" << std::endl;
    
    // Проверяем API режим
    bool api_mode = false;
    for(int i = 1; i < argc; i++) {
        if(std::string(argv[i]) == "--api") {
            api_mode = true;
            break;
        }
    }
    
    std::cout << "DEBUG: API mode = " << (api_mode ? "true" : "false") << std::endl;
    
    if(api_mode) {
        std::cout << "DEBUG: Entering API mode" << std::endl;
        
        // Простой API режим
        std::string user_input;
        std::cout << "DEBUG: Waiting for input..." << std::endl;
        
        if(std::getline(std::cin, user_input)) {
            std::cout << "DEBUG: Received input: '" << user_input << "'" << std::endl;
            
            // Простые ответы
            std::transform(user_input.begin(), user_input.end(), user_input.begin(), ::tolower);
            
            if(user_input.find("привет") != std::string::npos) {
                std::cout << "Neural Network: Привет! Как дела?" << std::endl;
            } else if(user_input.find("hello") != std::string::npos) {
                std::cout << "Neural Network: Hello! How are you?" << std::endl;
            } else if(user_input.find("как дела") != std::string::npos) {
                std::cout << "Neural Network: У меня все отлично! А у тебя?" << std::endl;
            } else if(user_input.find("how are you") != std::string::npos) {
                std::cout << "Neural Network: I'm doing great! And you?" << std::endl;
            } else {
                std::cout << "Neural Network: That's interesting! Tell me more." << std::endl;
            }
        } else {
            std::cout << "DEBUG: Failed to read input" << std::endl;
        }
    } else {
        std::cout << "DEBUG: Interactive mode" << std::endl;
        std::cout << "=== Simple C++ Neural Network ===" << std::endl;
    }
    
    std::cout << "DEBUG: Program ending" << std::endl;
    return 0;
}

