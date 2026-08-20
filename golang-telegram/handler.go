package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"
)

type TelegramUpdate struct {
	UpdateID int
	Message  *TelegramMessage
}

type TelegramMessage struct {
	MessageID int64
	Chat      TelegramChat
	From      *TelegramUser
	Text      string
}

type TelegramChat struct {
	ID int64
}

type TelegramUser struct {
	ID        int64
	FirstName string
	UserName  string
}

// QueryJob is the payload pushed to the Redis query stream.
type QueryJob struct {
	JobID            string    `json:"job_id"`
	TelegramUpdateID int       `json:"telegram_update_id"`
	Type             string    `json:"type"`
	ChatID           int64     `json:"chat_id"`
	Text             string    `json:"text"`
	ReceivedAt       time.Time `json:"received_at"`
}

// IngestJob is the payload pushed to the Redis ingest stream.
type IngestJob struct {
	JobID             string    `json:"job_id"`
	TelegramUpdateID  int       `json:"telegram_update_id"`
	Type              string    `json:"type"`
	ChatID            int64     `json:"chat_id"`
	SenderName        string    `json:"sender_name"`
	Text              string    `json:"text"`
	TelegramMessageID int64     `json:"telegram_message_id"`
	SkipGrading       bool      `json:"skip_grading"`
	ReceivedAt        time.Time `json:"received_at"`
}

var allowedUserID int64

func handleUpdate(ctx context.Context, update TelegramUpdate, queue *Queue) error {
	// The Go bridge only authorizes and routes; all grading/search work happens downstream.
	if update.Message == nil {
		return nil
	}

	if update.Message.From == nil || update.Message.From.ID != allowedUserID {
		return nil
	}

	route := classify(update.Message.Text)

	if route == "query" {
		job := QueryJob{
			JobID:            fmt.Sprintf("%d", time.Now().UnixNano()),
			TelegramUpdateID: update.UpdateID,
			Type:             "query",
			ChatID:           update.Message.Chat.ID,
			Text:             cleanText(update.Message.Text),
			ReceivedAt:       time.Now().UTC(),
		}

		payload, err := json.Marshal(job)
		if err != nil {
			return err
		}

		if err := queue.Publish(ctx, "manthan:queries", string(payload)); err != nil {
			return err
		}

		log.Println("published query job:", job.JobID)
	} else {
		content, skip := parseIngest(update.Message.Text)
		via, content := parseVia(content)
		job := IngestJob{
			JobID:             fmt.Sprintf("%d", time.Now().UnixNano()),
			TelegramUpdateID:  update.UpdateID,
			Type:              "ingest",
			ChatID:            update.Message.Chat.ID,
			SenderName:        resolveSender(update.Message.From, via),
			Text:              content,
			TelegramMessageID: update.Message.MessageID,
			SkipGrading:       skip,
			ReceivedAt:        time.Now().UTC(),
		}

		payload, err := json.Marshal(job)
		if err != nil {
			return err
		}

		if err := queue.Publish(ctx, "manthan:ingest", string(payload)); err != nil {
			return err
		}

		log.Println("published ingest job:", job.JobID)
	}

	return nil
}

func classify(text string) string {
	// Prefix-based routing keeps Telegram usage simple without a separate command parser.
	text = strings.TrimSpace(text)

	if strings.HasPrefix(text, "/ask") {
		return "query"
	}

	if strings.HasPrefix(text, "/ingest") {
		return "ingest"
	}

	if strings.HasPrefix(text, "/via") {
		return "ingest"
	}

	if strings.Contains(text, "http://") || strings.Contains(text, "https://") {
		return "ingest"
	}

	return "query"
}

func cleanText(text string) string {
	text = strings.TrimSpace(text)

	if strings.HasPrefix(text, "/ask") {
		return strings.TrimSpace(strings.TrimPrefix(text, "/ask"))
	}

	return text
}

func parseIngest(text string) (content string, skipGrading bool) {
	// "--trusted" lets the operator bypass grading for already-vetted notes.
	content = strings.TrimSpace(text)

	if strings.HasPrefix(content, "/ingest") {
		content = strings.TrimSpace(strings.TrimPrefix(content, "/ingest"))
	}

	if strings.HasPrefix(content, "--trusted") {
		skipGrading = true
		content = strings.TrimSpace(strings.TrimPrefix(content, "--trusted"))
	}

	return content, skipGrading
}

func parseVia(text string) (via string, rest string) {
	// "/via Name: ..." preserves the original speaker when forwarding someone else's note.
	text = strings.TrimSpace(text)

	if !strings.HasPrefix(text, "/via") {
		return "", text
	}

	text = strings.TrimSpace(strings.TrimPrefix(text, "/via"))

	if i := strings.Index(text, ":"); i != -1 {
		return strings.TrimSpace(text[:i]), strings.TrimSpace(text[i+1:])
	}

	if i := strings.IndexAny(text, " \t"); i != -1 {
		return text[:i], strings.TrimSpace(text[i+1:])
	}

	return "", text
}

func resolveSender(from *TelegramUser, via string) string {
	if via != "" {
		return via
	}

	if from != nil {
		if from.FirstName != "" {
			return from.FirstName
		}
		return from.UserName
	}

	return ""
}
