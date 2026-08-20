package main

import (
	"bufio"
	"context"
	"log"
	"os"
	"strconv"
	"strings"
)

type Config struct {
	BotToken  string
	RedisAddr string
}

func main() {
	loadEnv(".env")

	allowedUserID = mustGetInt64("ALLOWED_TELEGRAM_USER_ID")

	arg := Config{
		BotToken:  mustGetEnv("TELEGRAM_BOT_TOKEN"),
		RedisAddr: mustGetEnv("REDIS_ADDR"),
	}

	ctx := context.Background()

	queue := NewQueue(arg.RedisAddr)
	defer queue.Close()

	if err := queue.Ping(ctx); err != nil {
		log.Fatal("redis connection failed:", err)
	}

	log.Println("Connected to Redis")

	log.Println("Starting Manthan Telegram polling")

	if err := runPolling(ctx, arg.BotToken, queue); err != nil {
		log.Fatal(err)
	}

}

func mustGetEnv(key string) string {
	value := os.Getenv(key)

	if value == "" {
		log.Fatalf("missing environment variable: %s", key)
	}

	return value
}

func mustGetInt64(key string) int64 {
	value := mustGetEnv(key)

	n, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		log.Fatalf("invalid %s: %q", key, value)
	}

	return n
}

func loadEnv(path string) {
	file, err := os.Open(path)
	if err != nil {
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		value := strings.Trim(strings.TrimSpace(parts[1]), `"`)

		if _, ok := os.LookupEnv(key); !ok {
			os.Setenv(key, value)
		}
	}
}
