from locust import HttpUser, task, between

class SearchLoadTest(HttpUser):
    wait_time = between(1, 3)
    
    host = "https://giva-software-similarity-search.onrender.com"  

    @task
    def search_query(self):
        self.client.get("/api/search?q=AI")
