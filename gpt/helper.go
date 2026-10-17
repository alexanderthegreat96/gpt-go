package gpt

// little helper to get the Backend Name
func GetActiveBackend() string {
	return activeBackend.GetBackendName()
}
