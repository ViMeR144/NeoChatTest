#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <map>
#include <algorithm>
#include <locale>

// Собственный Tensor класс
class Tensor {
public:
    std::vector<std::vector<float>> data;
    int rows, cols;
    
    Tensor(int r, int c) : rows(r), cols(c) {
        data.resize(r, std::vector<float>(c, 0.0f));
    }
    
    void randomInit() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0, 0.1);
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                data[i][j] = dist(gen);
            }
        }
    }
    
    Tensor matmul(const Tensor& other) {
        Tensor result(rows, other.cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < other.cols; j++) {
                for(int k = 0; k < cols; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }
    
    Tensor add(const Tensor& other) {
        Tensor result(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
    void relu() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                data[i][j] = std::max(0.0f, data[i][j]);
            }
        }
    }
    
    void softmax() {
        for(int i = 0; i < rows; i++) {
            float sum = 0.0f;
            for(int j = 0; j < cols; j++) {
                sum += std::exp(data[i][j]);
            }
            for(int j = 0; j < cols; j++) {
                data[i][j] = std::exp(data[i][j]) / sum;
            }
        }
    }
};

// Двуязычный чат-бот
class ChatBot {
private:
    std::map<std::string, std::string> responses;
    
public:
    ChatBot() {
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
};

int main(int argc, char* argv[]) {
    // Проверяем, запущена ли программа в API режиме
    bool api_mode = false;
    for(int i = 1; i < argc; i++) {
        if(std::string(argv[i]) == "--api") {
            api_mode = true;
            break;
        }
    }
    
    ChatBot bot;
    
    if(api_mode) {
        // API режим: читаем одну строку и выдаем ответ
        std::string user_input;
        try {
            if(std::getline(std::cin, user_input)) {
                std::string response = bot.generateResponse(user_input);
                std::cout << "Neural Network: " << response << std::endl;
            } else {
                std::cout << "Neural Network: Hello! How can I help you?" << std::endl;
            }
        } catch(const std::exception& e) {
            std::cout << "Neural Network: I had an error processing your message." << std::endl;
        }
    } else {
        // Интерактивный режим
        std::cout << "=== My Own C++ Neural Network ===" << std::endl;
        std::cout << "Type 'exit' to quit" << std::endl;
        std::cout << std::endl;
        
        std::string user_input;
        
        while(true) {
            std::cout << "You: ";
            std::getline(std::cin, user_input);
            
            if(user_input == "exit") {
                std::cout << "Neural Network: Goodbye!" << std::endl;
                break;
            }
            
            std::string response = bot.generateResponse(user_input);
            std::cout << "Neural Network: " << response << std::endl;
            std::cout << std::endl;
        }
    }
    
    return 0;
}