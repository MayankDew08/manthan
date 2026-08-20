package main

import (
	"context"

	"github.com/redis/go-redis/v9"
)

// Queue is the Redis Streams wrapper shared by the Telegram bridge.
type Queue struct {
	client *redis.Client
}

// NewQueue opens a client for the Redis address configured for the bridge.
func NewQueue(addr string) *Queue {
	client := redis.NewClient(&redis.Options{
		Addr: addr,
	})

	return &Queue{
		client: client,
	}
}

// Ping verifies Redis is reachable before the bot starts polling.
func (q *Queue) Ping(ctx context.Context) error {
	return q.client.Ping(ctx).Err()
}

// Publish appends a serialized job payload to the target Redis stream.
func (q *Queue) Publish(
	ctx context.Context,
	stream string,
	payload string,
) error {
	return q.client.XAdd(ctx, &redis.XAddArgs{
		Stream: stream,
		Values: map[string]interface{}{
			"data": payload,
		},
	}).Err()
}

// Close releases the Redis client owned by the bridge process.
func (q *Queue) Close() error {
	return q.client.Close()
}
