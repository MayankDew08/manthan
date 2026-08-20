package main

import "testing"

func TestClassify(t *testing.T) {
	tests := []struct {
		text string
		want string
	}{
		{"/ask what is qdrant?", "query"},
		{"/ingest useful note", "ingest"},
		{"/via Alice: useful note", "ingest"},
		{"https://example.com/article", "ingest"},
		{"what did we save?", "query"},
	}

	for _, test := range tests {
		if got := classify(test.text); got != test.want {
			t.Errorf("classify(%q) = %q, want %q", test.text, got, test.want)
		}
	}
}

func TestCommandParsing(t *testing.T) {
	if got := cleanText(" /ask  stored topic "); got != "stored topic" {
		t.Fatalf("cleanText returned %q", got)
	}

	content, trusted := parseIngest("/ingest --trusted useful note")
	if content != "useful note" || !trusted {
		t.Fatalf("parseIngest returned %q, %v", content, trusted)
	}

	via, rest := parseVia("/via Alice: useful note")
	if via != "Alice" || rest != "useful note" {
		t.Fatalf("parseVia returned %q, %q", via, rest)
	}
}

func TestResolveSender(t *testing.T) {
	user := &TelegramUser{FirstName: "Alice", UserName: "alice_user"}
	if got := resolveSender(user, "Editor"); got != "Editor" {
		t.Fatalf("explicit sender returned %q", got)
	}
	if got := resolveSender(user, ""); got != "Alice" {
		t.Fatalf("first name returned %q", got)
	}
	if got := resolveSender(&TelegramUser{UserName: "alice_user"}, ""); got != "alice_user" {
		t.Fatalf("username returned %q", got)
	}
}
