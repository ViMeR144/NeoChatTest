#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <sstream>

class SmartNeuralNetwork {
private:
    std::map<std::string, std::vector<std::string>> knowledge_base;
    std::vector<std::string> greetings;
    std::vector<std::string> questions;
    std::vector<std::string> topics;
    std::mt19937 rng;
    
public:
    SmartNeuralNetwork() : rng(std::random_device{}()) {
        initializeKnowledgeBase();
    }
    
    void initializeKnowledgeBase() {
        // Приветствия
        greetings = {
            "Привет! Как дела?",
            "Здравствуй! Рад тебя видеть!",
            "Привет! Как настроение?",
            "Добро пожаловать! Как поживаешь?",
            "Привет! Что нового?"
        };
        
        // Вопросы для диалога
        questions = {
            "А что ты думаешь об этом?",
            "Расскажи больше об этом.",
            "Это интересно! А как ты к этому пришел?",
            "Любопытно! Что тебя больше всего интересует?",
            "Здорово! А что еще тебя волнует?"
        };
        
        // Темы для разговора
        topics = {
            "программирование", "технологии", "наука", "искусство", "музыка", 
            "спорт", "путешествия", "еда", "книги", "фильмы", "игры"
        };
        
        // База знаний
        knowledge_base["программирование"] = {
            "Программирование - это искусство создания программ!",
            "C++ - мощный язык программирования.",
            "Алгоритмы - основа программирования.",
            "Код должен быть читаемым и эффективным."
        };
        
        knowledge_base["технологии"] = {
            "Технологии развиваются очень быстро!",
            "Искусственный интеллект меняет мир.",
            "Роботы становятся умнее с каждым днем.",
            "Будущее за автоматизацией."
        };
        
        knowledge_base["наука"] = {
            "Наука помогает понять мир!",
            "Эксперименты - основа научного познания.",
            "Теории объясняют наблюдаемые явления.",
            "Открытия меняют наше понимание реальности."
        };
        
        knowledge_base["искусство"] = {
            "Искусство - это выражение души!",
            "Творчество вдохновляет и мотивирует.",
            "Каждый видит искусство по-своему.",
            "Красота субъективна."
        };
        
        knowledge_base["музыка"] = {
            "Музыка - это язык эмоций!",
            "Мелодии могут поднять настроение.",
            "Разные жанры для разных настроений.",
            "Музыка объединяет людей."
        };
        
        knowledge_base["спорт"] = {
            "Спорт укрепляет тело и дух!",
            "Регулярные тренировки полезны для здоровья.",
            "Соревнования мотивируют к улучшениям.",
            "Командный дух важен в спорте."
        };
        
        knowledge_base["путешествия"] = {
            "Путешествия расширяют кругозор!",
            "Новые места дают новые впечатления.",
            "Культуры разных стран уникальны.",
            "Встречи с новыми людьми обогащают."
        };
        
        knowledge_base["еда"] = {
            "Хорошая еда радует душу!",
            "Кулинария - это творчество.",
            "Разные культуры - разные вкусы.",
            "Еда объединяет людей за столом."
        };
        
        knowledge_base["книги"] = {
            "Книги - источник знаний и мудрости!",
            "Чтение развивает воображение.",
            "Каждая книга - новое приключение.",
            "Книги учат думать и анализировать."
        };
        
        knowledge_base["фильмы"] = {
            "Фильмы переносят в другие миры!",
            "Кино - это искусство рассказывания историй.",
            "Разные жанры для разных настроений.",
            "Хорошие фильмы запоминаются навсегда."
        };
        
        knowledge_base["игры"] = {
            "Игры развивают логику и реакцию!",
            "Игровой процесс может быть очень увлекательным.",
            "Многопользовательские игры объединяют людей.",
            "Игры - это развлечение и обучение."
        };
    }
    
    std::string generateResponse(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // Анализ типа сообщения
        if (isGreeting(lower_input)) {
            return getRandomGreeting();
        }
        
        if (isQuestion(lower_input)) {
            return generateQuestionResponse(lower_input);
        }
        
        // Поиск ключевых слов
        std::vector<std::string> found_topics = findTopics(lower_input);
        if (!found_topics.empty()) {
            return generateTopicResponse(found_topics, lower_input);
        }
        
        // Генерация общего ответа
        return generateGeneralResponse(lower_input);
    }
    
private:
    bool isGreeting(const std::string& input) {
        std::vector<std::string> greeting_words = {
            "привет", "здравствуй", "добрый день", "добрый вечер", 
            "доброе утро", "салют", "хай", "hello", "hi", "hey"
        };
        
        for (const auto& word : greeting_words) {
            if (input.find(word) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    
    bool isQuestion(const std::string& input) {
        std::vector<std::string> question_words = {
            "что", "как", "где", "когда", "почему", "зачем", "кто", "какой", "какая"
        };
        
        return input.find("?") != std::string::npos || 
               std::any_of(question_words.begin(), question_words.end(),
                   [&input](const std::string& word) {
                       return input.find(word) != std::string::npos;
                   });
    }
    
    std::vector<std::string> findTopics(const std::string& input) {
        std::vector<std::string> found;
        
        for (const auto& topic : topics) {
            if (input.find(topic) != std::string::npos) {
                found.push_back(topic);
            }
        }
        
        return found;
    }
    
    std::string getRandomGreeting() {
        std::uniform_int_distribution<> dis(0, greetings.size() - 1);
        return greetings[dis(rng)];
    }
    
    std::string generateQuestionResponse(const std::string& input) {
        std::vector<std::string> responses = {
            "Отличный вопрос! Дай мне подумать...",
            "Интересно! А что тебя больше всего интересует в этом?",
            "Хороший вопрос! Расскажи, что ты об этом знаешь?",
            "Любопытно! А как ты сам к этому относишься?",
            "Интригующий вопрос! Что привело тебя к этому?"
        };
        
        std::uniform_int_distribution<> dis(0, responses.size() - 1);
        return responses[dis(rng)];
    }
    
    std::string generateTopicResponse(const std::vector<std::string>& topics, const std::string& input) {
        std::uniform_int_distribution<> topic_dis(0, topics.size() - 1);
        std::string selected_topic = topics[topic_dis(rng)];
        
        auto it = knowledge_base.find(selected_topic);
        if (it != knowledge_base.end()) {
            std::uniform_int_distribution<> fact_dis(0, it->second.size() - 1);
            std::string fact = it->second[fact_dis(rng)];
            
            std::uniform_int_distribution<> question_dis(0, questions.size() - 1);
            std::string question = questions[question_dis(rng)];
            
            return fact + " " + question;
        }
        
        return generateGeneralResponse(input);
    }
    
    std::string generateGeneralResponse(const std::string& input) {
        std::vector<std::string> responses = {
            "Интересно! Расскажи мне больше об этом.",
            "Понятно! А что ты думаешь по этому поводу?",
            "Любопытно! Как ты к этому пришел?",
            "Здорово! А что еще тебя интересует?",
            "Отлично! Хочешь обсудить что-то еще?",
            "Классно! О чем еще поговорим?",
            "Прекрасно! Что тебя больше всего волнует?",
            "Замечательно! Есть ли что-то, о чем ты хотел бы поговорить?"
        };
        
        std::uniform_int_distribution<> dis(0, responses.size() - 1);
        return responses[dis(rng)];
    }
};

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
        // Умная нейросеть
        SmartNeuralNetwork neural;
        
        // Получаем сообщение как аргумент
        std::string user_input;
        
        // Ищем аргумент после --api
        for(int i = 1; i < argc; i++) {
            if(std::string(argv[i]) == "--api" && i + 1 < argc) {
                user_input = argv[i + 1];
                break;
            }
        }
        
        // Если нет аргумента, читаем из stdin
        if(user_input.empty()) {
            std::getline(std::cin, user_input);
        }
        
        // Генерируем умный ответ
        std::string response = neural.generateResponse(user_input);
        std::cout << "Neural Network: " << response << std::endl;
        
    } else {
        // Интерактивный режим
        std::cout << "=== Smart C++ Neural Network ===" << std::endl;
        std::cout << "Type 'exit' to quit" << std::endl;
        
        SmartNeuralNetwork neural;
        std::string user_input;
        
        while(true) {
            std::cout << "You: ";
            std::getline(std::cin, user_input);
            
            if(user_input == "exit") {
                std::cout << "Neural Network: Goodbye!" << std::endl;
                break;
            }
            
            std::string response = neural.generateResponse(user_input);
            std::cout << "Neural Network: " << response << std::endl;
        }
    }
    
    return 0;
}
