package handler

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
)

// 新增：存储agent模型ID的全局变量
var agentModelIDs []string

func init() {
	// 初始化随机数生成器
	rand.Seed(time.Now().UnixNano())

	// 新增：初始化agent模型ID
	initAgentModelIDs()
}

// 新增：初始化函数读取环境变量中的agent模型ID
func initAgentModelIDs() {
	agentModelIDsStr := os.Getenv("AGENT_MODEL_IDS")
	if agentModelIDsStr != "" {
		// 分割字符串，获取所有agent模型ID
		agentModelIDs = strings.Split(agentModelIDsStr, ",")
		// 去除每个ID的空白字符
		for i := range agentModelIDs {
			agentModelIDs[i] = strings.TrimSpace(agentModelIDs[i])
		}
		fmt.Printf("已加载 %d 个Agent模型ID: %v\n", len(agentModelIDs), agentModelIDs)
	} else {
		fmt.Println("未设置Agent模型ID环境变量，仅使用默认模型")
	}
}

// 新增：检查模型是否为agent模型
func isAgentModel(modelID string) bool {
	for _, id := range agentModelIDs {
		if id == modelID {
			return true
		}
	}
	return false
}

// TokenCount 定义了 token 计数的结构
type TokenCount struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

const (
	MaxContextTokens = 2000 // 最大上下文 token 数
)

// YouChatResponse 定义了从 You.com API 接收的单个 token 的结构。
type YouChatResponse struct {
	YouChatToken string `json:"youChatToken"`
}

// OpenAIStreamResponse 定义了 OpenAI API 流式响应的结构。
type OpenAIStreamResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}

// Choice 定义了 OpenAI 流式响应中 choices 数组的单个元素的结构。
type Choice struct {
	Delta        Delta  `json:"delta"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

// Delta 定义了流式响应中表示增量内容的结构。
type Delta struct {
	Content string `json:"content"`
}

// OpenAIRequest 定义了 OpenAI API 请求体的结构。
type OpenAIRequest struct {
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Model    string    `json:"model"`
}

// Message 定义了 OpenAI 聊天消息的结构，支持多模态内容。
type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // 可以是 string 或 []ContentPart
}

// ContentPart 定义了消息内容的一部分，支持文本和图片。
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL 定义了图片 URL 的结构。
type ImageURL struct {
	URL string `json:"url"`
}

// OpenAIResponse 定义了 OpenAI API 非流式响应的结构。
type OpenAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
}

// OpenAIChoice 定义了 OpenAI 非流式响应中 choices 数组的单个元素的结构。
type OpenAIChoice struct {
	Message      Message `json:"message"`
	Index        int     `json:"index"`
	FinishReason string  `json:"finish_reason"`
}

// ModelResponse 定义了 /v1/models 响应的结构。
type ModelResponse struct {
	Object string        `json:"object"`
	Data   []ModelDetail `json:"data"`
}

// ModelDetail 定义了模型列表中单个模型的详细信息。
type ModelDetail struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// modelMap 存储 OpenAI 模型名称到 You.com 模型名称的映射。
var modelMap = map[string]string{
	"deepseek-reasoner":       "deepseek_r1",
	"deepseek-chat":           "deepseek_v3",
	"o3-mini-high":            "openai_o3_mini_high",
	"o1":                      "openai_o1",
	"gpt-4o":                  "gpt_4o",
	"gpt-4o-mini":             "gpt_4o_mini",
	"claude-3-opus":           "claude_3_opus",
	"claude-3.5-sonnet":       "claude_3_5_sonnet",
	"gemini-1.5-pro":          "gemini_1_5_pro",
	"gemini-2.0-flash":        "gemini_2_flash",
	"llama-3.3-70b":           "llama3_3_70b",
	"llama-4-maverick":        "llama4_maverick",
	"llama-4-scout":           "llama4_scout",
	"mistral-large-2":         "mistral_large_2",
	"qwq-32b":                 "qwq_32b",
	"qwen-2.5-72b":            "qwen2p5_72b",
	"qwen-2.5-coder-32b":      "qwen2p5_coder_32b",
	"command-r-plus":          "command_r_plus",
	"claude-3-7-sonnet":       "claude_3_7_sonnet",
	"claude-3-7-sonnet-think": "claude_3_7_sonnet_thinking",
        "claude-4-sonnet":         "claude_4_sonnet",
	"claude-4-sonnet-think":   "claude_4_sonnet_thinking",
        "claude-4-opus":           "claude_4_opus",
	"claude-4-opus-think":     "claude_4_opus_thinking",
	"gemini-2.5-pro":	   "gemini_2_5_pro_preview",
	"o3":	   "openai_o3",
	"o3-pro":	   "openai_o3_pro",
	"o4-mini-high":	   "openai_o4_mini_high",
	"gpt-4.1":	   "gpt_4_1",
	"gpt-4.1-mini":	   "gpt_4_1_mini",
	"grok-4":	   "grok_4",
	"grok-3-beta":	   "grok_3",
	"grok-2":	   "grok_2",
	"grok-3-mini-beta":	   "grok_3_mini",
        "nous-hermes-2":	   "nous_hermes_2",
}

// getReverseModelMap 创建并返回 modelMap 的反向映射（You.com 模型名称 -> OpenAI 模型名称）。
func getReverseModelMap() map[string]string {
	reverse := make(map[string]string, len(modelMap))
	for k, v := range modelMap {
		reverse[v] = k
	}
	return reverse
}

// mapModelName 将 OpenAI 模型名称映射到 You.com 模型名称。
func mapModelName(openAIModel string) string {
	if mappedModel, exists := modelMap[openAIModel]; exists {
		return mappedModel
	}
	return "deepseek_v3" // 默认模型
}

// reverseMapModelName 将 You.com 模型名称映射回 OpenAI 模型名称。
func reverseMapModelName(youModel string) string {
	reverseMap := getReverseModelMap()
	if mappedModel, exists := reverseMap[youModel]; exists {
		return mappedModel
	}
	return "deepseek-chat" // 默认模型
}

// originalModel 存储原始的 OpenAI 模型名称。
var originalModel string

// NonceResponse 定义了获取 nonce 的响应结构
type NonceResponse struct {
	Uuid string
}

// UploadResponse 定义了文件上传的响应结构
type UploadResponse struct {
	Filename     string `json:"filename"`
	UserFilename string `json:"user_filename"`
}

// 定义最大查询长度
const MaxQueryLength = 2000

// ChatEntry 定义了聊天历史中的单个问答对的结构
type ChatEntry struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// extractTextContent 从消息内容中提取文本部分
func extractTextContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var textParts []string
		for _, part := range v {
			if partMap, ok := part.(map[string]interface{}); ok {
				if partType, exists := partMap["type"]; exists && partType == "text" {
					if text, exists := partMap["text"]; exists {
						if textStr, ok := text.(string); ok {
							textParts = append(textParts, textStr)
						}
					}
				}
			}
		}
		return strings.Join(textParts, " ")
	default:
		return ""
	}
}

// hasImageContent 检查消息是否包含图片内容
func hasImageContent(content interface{}) bool {
	switch v := content.(type) {
	case []interface{}:
		for _, part := range v {
			if partMap, ok := part.(map[string]interface{}); ok {
				if partType, exists := partMap["type"]; exists && partType == "image_url" {
					return true
				}
			}
		}
	}
	return false
}

// processImageContent 处理消息中的图片内容，上传图片并返回处理后的文本和sources
func processImageContent(content interface{}, dsToken string) (string, string, error) {
	textContent := extractTextContent(content)
	
	if !hasImageContent(content) {
		return textContent, "", nil
	}
	
	var sources []map[string]interface{}
	
	switch v := content.(type) {
	case []interface{}:
		for _, part := range v {
			if partMap, ok := part.(map[string]interface{}); ok {
				if partType, exists := partMap["type"]; exists && partType == "image_url" {
					if imageURL, exists := partMap["image_url"]; exists {
						if imageURLMap, ok := imageURL.(map[string]interface{}); ok {
							if url, exists := imageURLMap["url"]; exists {
								if urlStr, ok := url.(string); ok {
									filename, userFilename, err := processImage(urlStr, dsToken)
									if err != nil {
										return "", "", fmt.Errorf("处理图片失败: %v", err)
									}
									
									// 添加到sources
									source := map[string]interface{}{
										"filename":      filename,
										"size_bytes":    0,
										"source_type":   "user_file",
										"user_filename": userFilename,
									}
									sources = append(sources, source)
									
									// 在文本中添加图片引用
									textContent += fmt.Sprintf("\n查看这个图片文件：%s", userFilename)
								}
							}
						}
					}
				}
			}
		}
	}
	
	// 将sources转换为JSON字符串
	sourcesJSON := ""
	if len(sources) > 0 {
		if sourcesBytes, err := json.Marshal(sources); err == nil {
			sourcesJSON = string(sourcesBytes)
		}
	}
	
	return textContent, sourcesJSON, nil
}

// processImage 处理单个图片（base64或URL），返回上传后的文件名
func processImage(imageURL, dsToken string) (string, string, error) {
	var imageData []byte
	var err error
	var originalFilename string
	
	if strings.HasPrefix(imageURL, "data:image/") {
		// 处理base64图片
		imageData, originalFilename, err = decodeBase64Image(imageURL)
		if err != nil {
			return "", "", fmt.Errorf("解码base64图片失败: %v", err)
		}
	} else {
		// 处理URL图片
		imageData, originalFilename, err = downloadImage(imageURL)
		if err != nil {
			return "", "", fmt.Errorf("下载图片失败: %v", err)
		}
	}
	
	// 创建临时文件
	tempFile, err := os.CreateTemp("", "upload_*"+filepath.Ext(originalFilename))
	if err != nil {
		return "", "", fmt.Errorf("创建临时文件失败: %v", err)
	}
	defer os.Remove(tempFile.Name())
	defer tempFile.Close()
	
	// 写入图片数据
	if _, err := tempFile.Write(imageData); err != nil {
		return "", "", fmt.Errorf("写入临时文件失败: %v", err)
	}
	tempFile.Close()
	
	// 上传文件
	uploadResp, err := uploadFile(dsToken, tempFile.Name())
	if err != nil {
		return "", "", fmt.Errorf("上传文件失败: %v", err)
	}
	
	return uploadResp.Filename, uploadResp.UserFilename, nil
}

// decodeBase64Image 解码base64图片数据
func decodeBase64Image(dataURL string) ([]byte, string, error) {
	// 解析data URL格式: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
	parts := strings.Split(dataURL, ",")
	if len(parts) != 2 {
		return nil, "", fmt.Errorf("无效的data URL格式")
	}
	
	// 提取MIME类型
	header := parts[0]
	var extension string
	if strings.Contains(header, "image/png") {
		extension = ".png"
	} else if strings.Contains(header, "image/jpeg") || strings.Contains(header, "image/jpg") {
		extension = ".jpg"
	} else if strings.Contains(header, "image/gif") {
		extension = ".gif"
	} else if strings.Contains(header, "image/webp") {
		extension = ".webp"
	} else {
		extension = ".png" // 默认
	}
	
	// 解码base64数据
	imageData, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, "", fmt.Errorf("base64解码失败: %v", err)
	}
	
	// 生成随机文件名
	filename := fmt.Sprintf("%s_image%s", generateRandomString(6), extension)
	
	return imageData, filename, nil
}

// downloadImage 从URL下载图片
func downloadImage(imageURL string) ([]byte, string, error) {
	resp, err := http.Get(imageURL)
	if err != nil {
		return nil, "", fmt.Errorf("下载图片失败: %v", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("下载图片失败，状态码: %d", resp.StatusCode)
	}
	
	imageData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("读取图片数据失败: %v", err)
	}
	
	// 从URL中提取文件名
	filename := filepath.Base(imageURL)
	if filename == "." || filename == "/" {
		filename = "downloaded_image.jpg"
	}
	
	return imageData, filename, nil
}

// generateRandomString 生成指定长度的随机字符串
func generateRandomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[rand.Intn(len(charset))]
	}
	return string(b)
}

// countTokens 计算消息的token数量，包括图片token估算
func countTokens(content interface{}) int {
	textContent := extractTextContent(content)
	baseTokens := len(strings.Fields(textContent)) // 简单的单词计数估算
	
	// 如果包含图片，每张图片估算85个token
	if hasImageContent(content) {
		imageCount := 0
		if v, ok := content.([]interface{}); ok {
			for _, part := range v {
				if partMap, ok := part.(map[string]interface{}); ok {
					if partType, exists := partMap["type"]; exists && partType == "image_url" {
						imageCount++
					}
				}
			}
		}
		baseTokens += imageCount * 85
	}
	
	return baseTokens
}

// convertSystemToUser 将system消息转换为user消息，提取文本内容
func convertSystemToUser(message Message) Message {
	return Message{
		Role:    "user",
		Content: extractTextContent(message.Content),
	}
}

// Handler 是处理所有传入 HTTP 请求的主处理函数。
func Handler(w http.ResponseWriter, r *http.Request) {
	// 处理 /v1/models 请求（列出可用模型）
	if r.URL.Path == "/v1/models" || r.URL.Path == "/api/v1/models" {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		models := make([]ModelDetail, 0, len(modelMap))
		created := time.Now().Unix()
		for modelID := range modelMap {
			models = append(models, ModelDetail{
				ID:      modelID,
				Object:  "model",
				Created: created,
				OwnedBy: "organization-owner",
			})
		}

		// 新增：添加agent模型到模型列表
		for _, agentID := range agentModelIDs {
			models = append(models, ModelDetail{
				ID:      agentID,
				Object:  "model",
				Created: created,
				OwnedBy: "organization-owner",
			})
		}

		response := ModelResponse{
			Object: "list",
			Data:   models,
		}

		json.NewEncoder(w).Encode(response)
		return
	}

	// 处理非 /v1/chat/completions 请求（服务状态检查）
	if r.URL.Path != "/v1/chat/completions" && r.URL.Path != "/none/v1/chat/completions" && r.URL.Path != "/such/chat/completions" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "You2Api Service Running...",
			"message": "MoLoveSze...",
		})
		return
	}

	// 设置 CORS 头部
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "*")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// 验证 Authorization 头部
	authHeader := r.Header.Get("Authorization")
	if !strings.HasPrefix(authHeader, "Bearer ") {
		http.Error(w, "Missing or invalid authorization header", http.StatusUnauthorized)
		return
	}
	dsToken := strings.TrimPrefix(authHeader, "Bearer ") // 提取 DS token

	// 解析 OpenAI 请求体
	var openAIReq OpenAIRequest
	if err := json.NewDecoder(r.Body).Decode(&openAIReq); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	originalModel = openAIReq.Model

	// 转换 system 消息为 user 消息
	openAIReq.Messages = convertSystemToUserMessages(openAIReq.Messages)

	// 打印OpenAI消息
	fmt.Printf("\n=== 接收到的OpenAI消息 ===\n")
	for i, msg := range openAIReq.Messages {
		textContent := extractTextContent(msg.Content)
		hasImages := hasImageContent(msg.Content)
		if hasImages {
			fmt.Printf("消息 %d: 角色=%s, 文本内容=%s [包含图片]\n", i, msg.Role, textContent)
		} else {
			fmt.Printf("消息 %d: 角色=%s, 内容=%s\n", i, msg.Role, textContent)
		}
	}
	fmt.Printf("===================\n\n")

	// 构建 You.com 聊天历史
	var chatHistory []ChatEntry
	var sources []map[string]interface{}

	// 处理历史消息（不包括最后一条）
	var currentQuestion string
	var currentAnswer string
	var hasQuestion bool
	var hasAnswer bool

	for i := 0; i < len(openAIReq.Messages)-1; i++ {
		msg := openAIReq.Messages[i]

		if msg.Role == "user" {
			// 如果已经有问题和回答，添加到历史
			if hasQuestion && hasAnswer {
				chatHistory = append(chatHistory, ChatEntry{
					Question: currentQuestion,
					Answer:   currentAnswer,
				})
				// 重置状态
				currentQuestion = extractTextContent(msg.Content)
				currentAnswer = ""
				hasQuestion = true
				hasAnswer = false
			} else if hasQuestion {
				// 如果已经有问题但没有回答，合并问题
				currentQuestion += "\n" + extractTextContent(msg.Content)
			} else {
				// 新的问题
				currentQuestion = extractTextContent(msg.Content)
				hasQuestion = true
			}
		} else if msg.Role == "assistant" {
			if hasQuestion {
				// 如果有问题，设置回答
				currentAnswer = extractTextContent(msg.Content)
				hasAnswer = true
			} else if hasAnswer {
				// 如果已经有回答但没有问题，合并回答
				currentAnswer += "\n" + extractTextContent(msg.Content)
			} else {
				// 没有问题的回答，创建空问题
				currentQuestion = ""
				currentAnswer = extractTextContent(msg.Content)
				hasQuestion = true
				hasAnswer = true
			}
		}
	}

	// 添加最后一对问答（如果有）
	if hasQuestion {
		if hasAnswer {
			chatHistory = append(chatHistory, ChatEntry{
				Question: currentQuestion,
				Answer:   currentAnswer,
			})
		} else {
			// 有问题但没有回答，添加空回答
			chatHistory = append(chatHistory, ChatEntry{
				Question: currentQuestion,
				Answer:   "",
			})
		}
	}

	// 处理聊天历史中的每个条目，上传文件
	for i := range chatHistory {
		entry := &chatHistory[i]

		// 处理问题
		if entry.Question != "" {
			questionTokenCount, _ := countTokensForMessages([]Message{{Role: "user", Content: entry.Question}})

			// 如果问题较长，上传为文件
			if questionTokenCount >= 30 {
				// 获取nonce
				_, err := getNonce(dsToken)
				if err != nil {
					fmt.Printf("获取nonce失败: %v\n", err)
					http.Error(w, "Failed to get nonce", http.StatusInternalServerError)
					return
				}

				// 创建问题临时文件
				questionShortFileName := generateShortFileName()
				questionTempFile := questionShortFileName + ".txt"

				if err := os.WriteFile(questionTempFile, addUTF8BOM(entry.Question), 0644); err != nil {
					fmt.Printf("创建问题文件失败: %v\n", err)
					http.Error(w, "Failed to create temp file", http.StatusInternalServerError)
					return
				}
				defer os.Remove(questionTempFile)

				// 上传问题文件
				questionUploadResp, err := uploadFile(dsToken, questionTempFile)
				if err != nil {
					fmt.Printf("上传问题文件失败: %v\n", err)
					http.Error(w, "Failed to upload file", http.StatusInternalServerError)
					return
				}

				// 添加问题文件源信息
				sources = append(sources, map[string]interface{}{
					"source_type":   "user_file",
					"filename":      questionUploadResp.Filename,
					"user_filename": questionUploadResp.UserFilename,
					"size_bytes":    len(entry.Question),
				})

				// 更新问题为文件引用
				entry.Question = fmt.Sprintf("查看这个文件并且直接与文件内容进行聊天：%s.txt", strings.TrimSuffix(questionUploadResp.UserFilename, ".txt"))
			}
		}

		// 处理回答 - 只对长回答进行文件上传
		if entry.Answer != "" {
			answerTokenCount, _ := countTokensForMessages([]Message{{Role: "assistant", Content: entry.Answer}})
			
			// 如果回答较长，上传为文件
			if answerTokenCount >= 30 {
				// 获取nonce
				_, err := getNonce(dsToken)
				if err != nil {
					fmt.Printf("获取nonce失败: %v\n", err)
					http.Error(w, "Failed to get nonce", http.StatusInternalServerError)
					return
				}

				// 创建回答临时文件
				answerShortFileName := generateShortFileName()
				answerTempFile := answerShortFileName + ".txt"

				if err := os.WriteFile(answerTempFile, addUTF8BOM(entry.Answer), 0644); err != nil {
					fmt.Printf("创建回答文件失败: %v\n", err)
					http.Error(w, "Failed to create temp file", http.StatusInternalServerError)
					return
				}
				defer os.Remove(answerTempFile)

				// 上传回答文件
				answerUploadResp, err := uploadFile(dsToken, answerTempFile)
				if err != nil {
					fmt.Printf("上传回答文件失败: %v\n", err)
					http.Error(w, "Failed to upload file", http.StatusInternalServerError)
					return
				}

				// 添加回答文件源信息
				sources = append(sources, map[string]interface{}{
					"source_type":   "user_file",
					"filename":      answerUploadResp.Filename,
					"user_filename": answerUploadResp.UserFilename,
					"size_bytes":    len(entry.Answer),
				})

				// 更新回答为文件引用
				entry.Answer = fmt.Sprintf("查看这个文件并且直接与文件内容进行聊天：%s.txt", strings.TrimSuffix(answerUploadResp.UserFilename, ".txt"))
			}
		}
	}

	// 输出构建的聊天历史
	fmt.Printf("聊天历史构建完成，共 %d 条记录\n", len(chatHistory))
	for i, entry := range chatHistory {
		fmt.Printf("历史 %d: Q=%s, A=%s\n", i, entry.Question, entry.Answer)
	}

	chatHistoryJSON, _ := json.Marshal(chatHistory)

	// 创建 You.com API 请求
	youReq, _ := http.NewRequest("GET", "https://you.com/api/streamingSearch", nil)

	// 生成必要的 ID
	chatId := uuid.New().String()
	conversationTurnId := uuid.New().String()
	traceId := fmt.Sprintf("%s|%s|%s", chatId, conversationTurnId, time.Now().Format(time.RFC3339))

	// 处理最后一条消息
	lastMessage := openAIReq.Messages[len(openAIReq.Messages)-1]
	lastMessageTokens, err := countTokensForMessages([]Message{lastMessage})
	if err != nil {
		http.Error(w, "Failed to count tokens", http.StatusInternalServerError)
		return
	}

	// 构建查询参数
	q := youReq.URL.Query()

	// 设置基本参数
	q.Add("page", "1")
	q.Add("count", "10")
	q.Add("safeSearch", "Moderate")
	q.Add("mkt", "zh-HK")
	q.Add("enable_worklow_generation_ux", "true")
	q.Add("domain", "youchat")
	q.Add("use_personalization_extraction", "true")
	q.Add("queryTraceId", chatId)
	q.Add("chatId", chatId)
	q.Add("conversationTurnId", conversationTurnId)
	q.Add("pastChatLength", fmt.Sprintf("%d", len(chatHistory)))
	//q.Add("selectedChatMode", "custom")
	//q.Add("selectedAiModel", mapModelName(openAIReq.Model))
	q.Add("enable_agent_clarification_questions", "true")
	q.Add("traceId", traceId)
	q.Add("use_nested_youchat_updates", "true")

	// 新增：根据模型类型设置不同的参数
	isAgent := isAgentModel(openAIReq.Model)
	if isAgent {
		// 新增：Agent模型: 只使用selectedChatMode=agent模型ID
		fmt.Printf("使用Agent模型: %s\n", openAIReq.Model)
		q.Add("selectedChatMode", openAIReq.Model) // 修改：直接使用模型ID作为chatMode
	} else {
		// 修改：默认模型: 使用selectedAiModel和selectedChatMode=custom
		fmt.Printf("使用默认模型: %s (映射为: %s)\n", openAIReq.Model, mapModelName(openAIReq.Model))
		q.Add("selectedAiModel", mapModelName(openAIReq.Model))
		q.Add("selectedChatMode", "custom")
	}

	// 处理最后一条消息的内容（包括图片）
	var finalQuery string
	var imageSources string
	
	// 检查是否包含图片内容
	if hasImageContent(lastMessage.Content) {
		// 处理图片内容
		textContent, imageSourcesJSON, err := processImageContent(lastMessage.Content, dsToken)
		if err != nil {
			fmt.Printf("处理图片内容失败: %v\n", err)
			http.Error(w, "Failed to process image content", http.StatusInternalServerError)
			return
		}
		finalQuery = textContent
		imageSources = imageSourcesJSON
	} else {
		// 纯文本内容
		finalQuery = extractTextContent(lastMessage.Content)
	}

	// 如果最后一条消息超过限制，使用文件上传
	if lastMessageTokens > MaxContextTokens {
		// 获取 nonce - 不再需要nonce
		_, err := getNonce(dsToken) // 仍然调用以保持API流程，但不使用返回值
		if err != nil {
			fmt.Printf("获取 nonce 失败: %v\n", err)
			http.Error(w, "Failed to get nonce", http.StatusInternalServerError)
			return
		}

		// 创建临时文件，使用短文件名
		shortFileName := generateShortFileName()
		tempFile := shortFileName + ".txt"

		// 确保使用UTF-8编码写入文件，添加BOM标记
		if err := os.WriteFile(tempFile, addUTF8BOM(finalQuery), 0644); err != nil {
			fmt.Printf("创建临时文件失败: %v\n", err)
			http.Error(w, "Failed to create temp file", http.StatusInternalServerError)
			return
		}
		defer os.Remove(tempFile)

		// 上传文件
		uploadResp, err := uploadFile(dsToken, tempFile)
		if err != nil {
			fmt.Printf("上传文件失败: %v\n", err)
			http.Error(w, "Failed to upload file", http.StatusInternalServerError)
			return
		}

		// 添加文件源信息
		sources = append(sources, map[string]interface{}{
			"source_type":   "user_file",
			"filename":      uploadResp.Filename,
			"user_filename": uploadResp.UserFilename,
			"size_bytes":    len(finalQuery),
		})

		// 使用文件引用作为查询，确保包含.txt后缀
		finalQuery = fmt.Sprintf("查看这个文件并且直接与文件内容进行聊天：%s.txt", strings.TrimSuffix(uploadResp.UserFilename, ".txt"))
	}

	// 合并图片sources和其他sources
	if imageSources != "" {
		var imageSourcesList []map[string]interface{}
		if err := json.Unmarshal([]byte(imageSources), &imageSourcesList); err == nil {
			sources = append(sources, imageSourcesList...)
		}
	}

	// 添加 sources 参数
	if len(sources) > 0 {
		sourcesJSON, _ := json.Marshal(sources)
		q.Add("sources", string(sourcesJSON))
	}

	q.Add("q", finalQuery)

	q.Add("chat", string(chatHistoryJSON))
	youReq.URL.RawQuery = q.Encode()

	// 添加调试信息
	fmt.Printf("\n=== 聊天历史内容 ===\n")
	fmt.Printf("历史条数: %d\n", len(chatHistory))
	for i, entry := range chatHistory {
		fmt.Printf("条目 %d:\n", i+1)
		fmt.Printf("  问题: %s\n", entry.Question)
		fmt.Printf("  回答: %s\n", entry.Answer)
	}
	fmt.Printf("chat参数内容: %s\n", string(chatHistoryJSON))
	fmt.Printf("===================\n\n")

	fmt.Printf("\n=== 完整请求信息 ===\n")
	fmt.Printf("请求 URL: %s\n", youReq.URL.String())
	fmt.Printf("请求头:\n")
	for key, values := range youReq.Header {
		fmt.Printf("%s: %v\n", key, values)
	}

	// 设置请求头
	youReq.Header = http.Header{
		"sec-ch-ua-platform":         {"Windows"},
		"Cache-Control":              {"no-cache"},
		"sec-ch-ua":                  {`"Not(A:Brand";v="99", "Microsoft Edge";v="133", "Chromium";v="133"`},
		"sec-ch-ua-bitness":          {"64"},
		"sec-ch-ua-model":            {""},
		"sec-ch-ua-mobile":           {"?0"},
		"sec-ch-ua-arch":             {"x86"},
		"sec-ch-ua-full-version":     {"133.0.3065.39"},
		"Accept":                     {"text/event-stream"},
		"User-Agent":                 {"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0"},
		"sec-ch-ua-platform-version": {"19.0.0"},
		"Sec-Fetch-Site":             {"same-origin"},
		"Sec-Fetch-Mode":             {"cors"},
		"Sec-Fetch-Dest":             {"empty"},
		"Host":                       {"you.com"},
	}

	// 设置 Cookie
	cookies := getCookies(dsToken)
	var cookieStrings []string
	for name, value := range cookies {
		cookieStrings = append(cookieStrings, fmt.Sprintf("%s=%s", name, value))
	}
	youReq.Header.Add("Cookie", strings.Join(cookieStrings, ";"))
	fmt.Printf("Cookie: %s\n", strings.Join(cookieStrings, ";"))
	fmt.Printf("===================\n\n")

	// 发送请求并获取响应
	client := &http.Client{}
	resp, err := client.Do(youReq)
	if err != nil {
		fmt.Printf("发送请求失败: %v\n", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// 打印响应状态码
	fmt.Printf("响应状态码: %d\n", resp.StatusCode)

	// 如果状态码不是 200，打印响应内容
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("错误响应内容: %s\n", string(body))
		http.Error(w, fmt.Sprintf("API returned status %d", resp.StatusCode), resp.StatusCode)
		return
	}

	// 根据 OpenAI 请求的 stream 参数选择处理函数
	if !openAIReq.Stream {
		handleNonStreamingResponse(w, youReq) // 处理非流式响应
		return
	}

	handleStreamingResponse(w, youReq) // 处理流式响应
}

// getCookies 根据提供的 DS token 生成所需的 Cookie。
func getCookies(dsToken string) map[string]string {
	return map[string]string{
		"guest_has_seen_legal_disclaimer": "true",
		"youchat_personalization":         "true",
		"DS":                              dsToken,                // 关键的 DS token
		"you_subscription":                "youpro_standard_year", // 示例订阅信息
		"youpro_subscription":             "true",
		"ai_model":                        "deepseek_r1", // 示例 AI 模型
		"youchat_smart_learn":             "true",
	}
}

// handleNonStreamingResponse 处理非流式请求。
func handleNonStreamingResponse(w http.ResponseWriter, youReq *http.Request) {
	client := &http.Client{
		Timeout: 60 * time.Second, // 设置超时时间
	}
	resp, err := client.Do(youReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	// 设置 scanner 的缓冲区大小（可选，但对于大型响应很重要）
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	// 逐行扫描响应，寻找 youChatToken 事件
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: youChatToken") {
			scanner.Scan() // 读取下一行 (data 行)
			data := scanner.Text()
			if !strings.HasPrefix(data, "data: ") {
				continue // 如果不是 data 行，则跳过
			}
			var token YouChatResponse
			if err := json.Unmarshal([]byte(strings.TrimPrefix(data, "data: ")), &token); err != nil {
				continue // 如果解析失败，则跳过
			}
			fullResponse.WriteString(token.YouChatToken) // 将 token 添加到完整响应中
		}
	}

	if scanner.Err() != nil {
		http.Error(w, "Error reading response", http.StatusInternalServerError)
		return
	}

	// 构建 OpenAI 格式的非流式响应
	openAIResp := OpenAIResponse{
		ID:      "chatcmpl-" + fmt.Sprintf("%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   reverseMapModelName(mapModelName(originalModel)), // 映射回 OpenAI 模型名称
		Choices: []OpenAIChoice{
			{
				Message: Message{
					Role:    "assistant",
					Content: fullResponse.String(), // 完整的响应内容
				},
				Index:        0,
				FinishReason: "stop", // 停止原因
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(openAIResp); err != nil {
		http.Error(w, "Error encoding response", http.StatusInternalServerError)
		return
	}
}

// handleStreamingResponse 处理流式请求。
func handleStreamingResponse(w http.ResponseWriter, youReq *http.Request) {
	client := &http.Client{} // 流式请求不需要设置超时，因为它会持续接收数据
	resp, err := client.Do(youReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// 设置流式响应的头部
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	scanner := bufio.NewScanner(resp.Body)
	// 逐行扫描响应，寻找 youChatToken 事件
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "event: youChatToken") {
			scanner.Scan()         // 读取下一行 (data 行)
			data := scanner.Text() // 获取数据行

			var token YouChatResponse
			json.Unmarshal([]byte(strings.TrimPrefix(data, "data: ")), &token) // 解析 JSON

			// 构建 OpenAI 格式的流式响应块
			openAIResp := OpenAIStreamResponse{
				ID:      "chatcmpl-" + fmt.Sprintf("%d", time.Now().Unix()),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   reverseMapModelName(mapModelName(originalModel)), // 映射回 OpenAI 模型名称
				Choices: []Choice{
					{
						Delta: Delta{
							Content: token.YouChatToken, // 增量内容
						},
						Index:        0,
						FinishReason: "", // 流式响应中通常为空
					},
				},
			}

			respBytes, _ := json.Marshal(openAIResp)          // 将响应块序列化为 JSON
			fmt.Fprintf(w, "data: %s\n\n", string(respBytes)) // 写入响应数据
			w.(http.Flusher).Flush()                          // 立即刷新输出
		}
	}

}

// 获取上传文件所需的 nonce
func getNonce(dsToken string) (*NonceResponse, error) {
	req, _ := http.NewRequest("GET", "https://you.com/api/get_nonce", nil)
	req.Header.Set("Cookie", fmt.Sprintf("DS=%s", dsToken))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// 读取完整的响应内容
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应失败: %v", err)
	}

	// 直接使用响应内容作为 UUID
	return &NonceResponse{
		Uuid: strings.TrimSpace(string(body)),
	}, nil
}

// 生成短文件名
func generateShortFileName() string {
	// 生成6位纯英文字母字符串
	const charset = "abcdefghijklmnopqrstuvwxyz"
	result := make([]byte, 6)
	for i := range result {
		result[i] = charset[rand.Intn(len(charset))]
	}
	return string(result)
}

// 上传文件
func uploadFile(dsToken, filePath string) (*UploadResponse, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return nil, err
	}

	if _, err := io.Copy(part, file); err != nil {
		return nil, err
	}
	writer.Close()

	req, _ := http.NewRequest("POST", "https://you.com/api/upload", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Cookie", fmt.Sprintf("DS=%s", dsToken))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// 打印上传响应状态码
	fmt.Printf("文件上传响应状态码: %d\n", resp.StatusCode)

	// 如果上传失败，记录错误响应
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		fmt.Printf("文件上传错误响应内容: %s\n", string(respBody))
		return nil, fmt.Errorf("上传文件失败，状态码: %d", resp.StatusCode)
	}

	// 先读取完整的响应体
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// 打印完整的响应内容
	fmt.Printf("文件上传响应内容: %s\n", string(respBody))

	// 解析响应
	var uploadResp UploadResponse
	if err := json.Unmarshal(respBody, &uploadResp); err != nil {
		return nil, err
	}

	// 打印解析后的响应
	fmt.Printf("上传文件成功: filename=%s, user_filename=%s\n",
		uploadResp.Filename, uploadResp.UserFilename)

	return &uploadResp, nil
}

// 计算消息的 token 数（使用字符估算方法），支持多模态内容
func countTokensForMessages(messages []Message) (int, error) {
	totalTokens := 0
	for _, msg := range messages {
		tokens := countTokens(msg.Content)
		totalTokens += tokens + 2 // 加上角色名的 token（约 2 个）
	}
	return totalTokens, nil
}

// 将 system 消息转换为第一条 user 消息，支持多模态内容
func convertSystemToUserMessages(messages []Message) []Message {
	if len(messages) == 0 {
		return messages
	}

	var systemContent strings.Builder
	var newMessages []Message
	var systemFound bool

	// 收集所有 system 消息
	for _, msg := range messages {
		if msg.Role == "system" {
			if systemContent.Len() > 0 {
				systemContent.WriteString("\n")
			}
			systemContent.WriteString(extractTextContent(msg.Content))
			systemFound = true
		} else {
			newMessages = append(newMessages, msg)
		}
	}

	// 如果有 system 消息，将其作为第一条 user 消息
	if systemFound {
		newMessages = append([]Message{{
			Role:    "user",
			Content: systemContent.String(),
		}}, newMessages...)
	}

	return newMessages
}

// 添加UTF-8 BOM标记的函数
func addUTF8BOM(content string) []byte {
	// 首先确保内容是纯文本
	content = ensurePlainText(content)
	// UTF-8 BOM: EF BB BF
	bom := []byte{0xEF, 0xBB, 0xBF}
	return append(bom, []byte(content)...)
}

// 确保内容为纯文本
func ensurePlainText(content string) string {
	// 移除可能导致问题的不可打印字符
	var result strings.Builder
	for _, r := range content {
		// 只保留ASCII可打印字符、基本中文字符和基本标点符号
		if (r >= 32 && r <= 126) || // ASCII可打印字符
			(r >= 0x4E00 && r <= 0x9FA5) || // 基本汉字
			(r >= 0x3000 && r <= 0x303F) || // 中文标点
			r == 0x000A || r == 0x000D { // 换行和回车
			result.WriteRune(r)
		} else {
			// 替换其他字符为空格
			result.WriteRune(' ')
		}
	}
	return result.String()
}


