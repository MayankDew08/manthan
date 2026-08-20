package main

import (
	"context"
	"log"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
)

// runPolling converts Telegram updates into the smaller internal routing model.
func runPolling(ctx context.Context, BotToken string, queue *Queue) error {
	bot, err := tgbotapi.NewBotAPI(BotToken)
	if err != nil {
		return err
	}

	log.Printf("Authorized as @%s", bot.Self.UserName)

	updateConfig := tgbotapi.NewUpdate(0)
	updateConfig.Timeout = 30

	updates := bot.GetUpdatesChan(updateConfig)

	for update := range updates {
		if update.Message == nil {
			continue
		}

		var from *TelegramUser
		if update.Message.From != nil {
			from = &TelegramUser{
				ID:        update.Message.From.ID,
				FirstName: update.Message.From.FirstName,
				UserName:  update.Message.From.UserName,
			}
		}

		ourUpdate := TelegramUpdate{
			UpdateID: update.UpdateID,
			Message: &TelegramMessage{
				MessageID: int64(update.Message.MessageID),
				Chat: TelegramChat{
					ID: update.Message.Chat.ID,
				},
				From: from,
				Text: update.Message.Text,
			},
		}

		if err := handleUpdate(ctx, ourUpdate, queue); err != nil {
			log.Printf("update %d failed: %v", update.UpdateID, err)
		}
	}

	return nil

}
