Run the iris_classifier service in production mode:

| Protocol | Command                                                  |
| -------- | -------------------------------------------------------- |
| HTTP     | `vtsserving serve-http iris_classifier:latest --production` |
| gRPC     | `vtsserving serve-grpc iris_classifier:latest --production` |

Start locust testing client:

```bash
locust --class-picker -H http://localhost:3000
```
