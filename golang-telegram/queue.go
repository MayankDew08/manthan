package main

import (
	"context"

	"github.com/redis/go-redis/v9"
)

type Queue struct {
	client *redis.Client
}

func NewQueue(addr string) *Queue {
	client := redis.NewClient(&redis.Options{
		Addr: addr,
	})

	return &Queue{
		client: client,
	}
}

func (q *Queue) Ping(ctx context.Context) error {
	return q.client.Ping(ctx).Err()
}

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

func (q *Queue) Close() error {
	return q.client.Close()
}
