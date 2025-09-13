#include <iostream>
#include <string>
#include <map>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <thread>
#include <chrono>

// Простой HTTP сервер для C++ нейросети
class SimpleHTTPServer {
private:
    int port;
    std::map<std::string, std::string> responses;
    
public:
    SimpleHTTPServer(int p) : port(p) {
        // Инициализируем ответы нейросети
        initializeResponses();
    }
    
    void initializeResponses() {
        // Русские ответы
        responses["привет"] = "Привет! Как дела?";
        responses["как дела"] = "У меня все отлично! А у тебя?";
        responses["что делаешь"] = "Я изучаю новые технологии!";
        responses["пока"] = "До свидания! Было приятно пообщаться!";
        responses["спасибо"] = "Пожалуйста! Рад помочь!";
        responses["как тебя зовут"] = "Я твоя собственная нейросеть!";
        responses["что ты умеешь"] = "Я могу общаться на русском и английском!";
        responses["хорошо"] = "Отлично! Рад это слышать!";
        responses["плохо"] = "Не расстраивайся! Все будет хорошо!";
        responses["как жизнь"] = "Жизнь прекрасна! Особенно когда есть такие собеседники как ты!";
        responses["что нового"] = "Много интересного! Я изучаю новые алгоритмы машинного обучения!";
        
        // Английские ответы
        responses["hello"] = "Hello! How are you?";
        responses["how are you"] = "I'm doing great! And you?";
        responses["what are you doing"] = "I'm studying new technologies!";
        responses["bye"] = "Goodbye! It was nice chatting!";
        responses["thank you"] = "You're welcome! Happy to help!";
        responses["what is your name"] = "I'm your own neural network!";
        responses["what can you do"] = "I can chat in Russian and English!";
        responses["good"] = "Great! I'm happy to hear that!";
        responses["bad"] = "Don't worry! Everything will be fine!";
        responses["how is life"] = "Life is wonderful! Especially with great friends like you!";
        responses["what's new"] = "Lots of interesting things! I'm learning new machine learning algorithms!";
        responses["im good"] = "That's wonderful! I'm glad you're doing well!";
        responses["i'm good"] = "That's wonderful! I'm glad you're doing well!";
        responses["fine"] = "Excellent! That makes me happy!";
        responses["great"] = "Fantastic! That's the spirit!";
        responses["awesome"] = "You're awesome too! Thanks for chatting with me!";
        
        // Дополнительные фразы
        responses["hi"] = "Hi there!";
        responses["good morning"] = "Good morning!";
        responses["good evening"] = "Good evening!";
        responses["доброе утро"] = "Доброе утро!";
        responses["добрый вечер"] = "Добрый вечер!";
        responses["yes"] = "I agree! That's a great point!";
        responses["no"] = "I understand. Sometimes things don't work out as planned.";
        responses["maybe"] = "That's a thoughtful response! Life is full of possibilities!";
    }
    
    std::string generateResponse(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // Специальные ответы для коротких фраз
        if(lower_input.length() <= 3) {
            if(lower_input == "ok" || lower_input == "ок") return "Okay! What else would you like to talk about?";
            if(lower_input == "hmm" || lower_input == "ммм") return "Thinking about something interesting?";
        }
        
        // Проверяем точные совпадения
        auto exact_match = responses.find(lower_input);
        if(exact_match != responses.end()) {
            return exact_match->second;
        }
        
        // Проверяем частичные совпадения
        for(const auto& pair : responses) {
            if(lower_input.find(pair.first) != std::string::npos) {
                return pair.second;
            }
        }
        
        // Разнообразные ответы для неизвестных фраз
        static std::vector<std::string> fallback_responses = {
            "That's fascinating! Tell me more about that.",
            "I find that very interesting! Can you elaborate?",
            "Wow, that sounds intriguing! What else can you tell me?",
            "That's a great topic! I'd love to hear more.",
            "Interesting perspective! What made you think of that?",
            "That's wonderful! I enjoy learning new things from you.",
            "How thoughtful! Tell me more about your thoughts on this.",
            "That's amazing! You always have such interesting things to say."
        };
        
        static int response_counter = 0;
        return fallback_responses[response_counter++ % fallback_responses.size()];
    }
    
    std::string urlDecode(const std::string& str) {
        std::string result;
        result.reserve(str.size());
        
        for (size_t i = 0; i < str.size(); ++i) {
            if (str[i] == '%' && i + 2 < str.size()) {
                int value;
                std::stringstream ss;
                ss << std::hex << str.substr(i + 1, 2);
                ss >> value;
                result += static_cast<char>(value);
                i += 2;
            } else if (str[i] == '+') {
                result += ' ';
            } else {
                result += str[i];
            }
        }
        
        return result;
    }
    
    std::string extractMessage(const std::string& request) {
        // Простой парсер для извлечения сообщения из POST запроса
        size_t body_start = request.find("\r\n\r\n");
        if (body_start == std::string::npos) {
            return "";
        }
        
        std::string body = request.substr(body_start + 4);
        
        // Ищем JSON с сообщением
        size_t messages_start = body.find("\"messages\"");
        if (messages_start == std::string::npos) {
            return "";
        }
        
        size_t content_start = body.find("\"content\":\"", messages_start);
        if (content_start == std::string::npos) {
            return "";
        }
        
        content_start += 11; // длина "\"content\":\""
        size_t content_end = body.find("\"", content_start);
        if (content_end == std::string::npos) {
            return "";
        }
        
        std::string content = body.substr(content_start, content_end - content_start);
        
        // Находим последнее сообщение пользователя
        size_t last_user = body.rfind("\"role\":\"user\"");
        if (last_user == std::string::npos || last_user < messages_start) {
            return "";
        }
        
        // Извлекаем контент последнего пользовательского сообщения
        size_t user_content_start = body.find("\"content\":\"", last_user);
        if (user_content_start == std::string::npos) {
            return "";
        }
        
        user_content_start += 11;
        size_t user_content_end = body.find("\"", user_content_start);
        if (user_content_end == std::string::npos) {
            return "";
        }
        
        return urlDecode(body.substr(user_content_start, user_content_end - user_content_start));
    }
    
    std::string createHTTPResponse(const std::string& body) {
        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n";
        response << "Content-Type: application/json\r\n";
        response << "Access-Control-Allow-Origin: *\r\n";
        response << "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n";
        response << "Access-Control-Allow-Headers: Content-Type\r\n";
        response << "Content-Length: " << body.length() << "\r\n";
        response << "\r\n";
        response << body;
        
        return response.str();
    }
    
    void startServer() {
        std::cout << "=== C++ Neural Network API Server ===" << std::endl;
        std::cout << "Starting server on port " << port << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        std::cout << std::endl;
        
        // Используем netcat для простого HTTP сервера
        std::string cmd = "echo 'Starting C++ Neural Network API Server...' && ";
        cmd += "while true; do ";
        cmd += "echo -e \"HTTP/1.1 200 OK\\r\\nContent-Type: application/json\\r\\nAccess-Control-Allow-Origin: *\\r\\n\\r\\n";
        cmd += "{\\\"ok\\\": true, \\\"reply\\\": \\\"C++ Neural Network API is running! Send POST requests to /chat\\\"}\" | ";
        cmd += "nc -l -p " + std::to_string(port) + " -q 1; ";
        cmd += "done";
        
        std::cout << "Run this command in another terminal to start the API server:" << std::endl;
        std::cout << cmd << std::endl;
        std::cout << std::endl;
        std::cout << "Or use this simpler approach:" << std::endl;
        std::cout << "python -m http.server " << port << " --bind 127.0.0.1" << std::endl;
    }
};

int main() {
    SimpleHTTPServer server(8080);
    server.startServer();
    return 0;
}