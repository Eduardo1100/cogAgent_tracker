The following code-block will be rendered as a Mermaid diagram:

```mermaid 
graph TD
    %% Define Styles and Colors
    classDef host fill:#455a64,stroke:#263238,stroke-width:2px,color:#fff,font-weight:bold;
    classDef interface fill:#eceff1,stroke:#cfd8dc,stroke-width:1px,color:#37474f,stroke-dasharray: 5 5;
    
    classDef containerApp fill:#009688,stroke:#00796b,stroke-width:2px,color:#fff,font-weight:bold;
    classDef containerDB fill:#1976d2,stroke:#1565c0,stroke-width:2px,color:#fff,font-weight:bold;
    classDef containerRedis fill:#d32f2f,stroke:#b71c1c,stroke-width:2px,color:#fff,font-weight:bold;
    classDef containerMinio fill:#ffa000,stroke:#ff8f00,stroke-width:2px,color:#263238,font-weight:bold;
    
    classDef volume fill:#eceff1,stroke:#cfd8dc,stroke-width:1px,color:#37474f;

    %% Diagram Structure
    subgraph Host_System [Host Machine]
        Host[.env File]:::host
        P8000((Port 8000)):::interface
        P5432((Port 5432)):::interface
        P9000((Port 9000)):::interface
        P9001((Port 9001)):::interface
    end

    subgraph Compose_Stack [Docker Compose Stack]
        App[app: cogagent-app]:::containerApp
        DB[(db: PostgreSQL 15)]:::containerDB
        Cache[(cache: Redis 7)]:::containerRedis
        Minio[objectstore: MinIO]:::containerMinio
    end

    subgraph Persistent_Volumes [Named Docker Volumes]
        VolVenv[venv_storage]:::volume
        VolUV[uv_cache]:::volume
        VolUVPy[uv_python]:::volume
        VolAlf[alfworld_data]:::volume
        VolWandb[wandb_data]:::volume
        VolDB[database-data]:::volume
        VolS3[objectstore-data]:::volume
    end

    %% Connections
    App -->|DB URL| DB
    App -->|Redis URL| Cache
    App -->|S3 URL| Minio

    App --- VolVenv
    App --- VolUV
    App --- VolUVPy
    App --- VolAlf
    App --- VolWandb
    DB --- VolDB
    Minio --- VolS3

    Host -.-> App
    Host -.-> DB
    Host -.-> Minio
    
    P8000 --- App
    P5432 --- DB
    P9000 --- Minio
    P9001 --- Minio
```