# Monitoring

Learn how to configure Prometheus & Grafana to monitor your Pinecone Projects with Docker Compose.

## Configuration

Each Project needs to be configured with its own Prometheus scraper. You will need a Pinecone API
Key to authorize Prometheus to fetch the metrics. For the target, you need to specify the Project's region, 
for example `us-west1-gcp`. Update the file found at `./prometheus/config.yml`

```yaml
scrape_configs:
  - job_name: pinecone-project-1
    authorization:
      credentials: <APIKEY>
    scheme: https
    static_configs:
      - targets: [ 'metrics.<PROJECT REGION>.pinecone.io' ]
```


### Grafana

Grafana is configured by default to run against the local Prometheus instance. A sample dashboard
for monitoring all indexes in a Project is preloaded into Grafana. You can add this dashboard to your own
Grafana instance by copying `./grafana/dashbords/pinecone.json`.

## Run

Start docker compose

```shell
docker-compose up
```

Check the ports Grafana and Promethus are running on

```shell
docker-compose ps
```

Open Grafana in your browser an navigate to Dashboard. The default dashboard is a folder titled Pinecone. 