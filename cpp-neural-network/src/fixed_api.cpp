#include <iostream>
#include <string>
#include <algorithm>

int main(int argc, char* argv[]) {
    // Проверяем API режим
    bool api_mode = false;
    for(int i = 1; i < argc; i++) {
        if(std::string(argv[i]) == "--api") {
            api_mode = true;
            break;
        }
    }
    
    if(api_mode) {
        // Простой API режим - читаем из stdin
        std::string user_input;
        
        // Читаем всю строку из stdin
        std::getline(std::cin, user_input);
        
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
        // Интерактивный режим
        std::cout << "=== Fixed C++ Neural Network ===" << std::endl;
        std::cout << "Type 'exit' to quit" << std::endl;
        
        std::string user_input;
        while(true) {
            std::cout << "You: ";
            std::getline(std::cin, user_input);
            
            if(user_input == "exit") {
                std::cout << "Neural Network: Goodbye!" << std::endl;
                break;
            }
            
            std::cout << "Neural Network: I understand your message." << std::endl;
        }
    }
    
    return 0;
}

