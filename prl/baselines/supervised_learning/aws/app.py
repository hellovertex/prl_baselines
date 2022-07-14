import json
import torch


def lambda_handler(event, context):
    model = torch.load("tmp/model.pt", map_location=torch.device('cpu'))
    model.eval()
    print("## EVENT ")
    print(event)

    print("## CONTEXT")
    print(context)
    obs = event['query']  # list[float]
    action = torch.argmax(model(torch.Tensor(obs)))

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps({
            'default': 'Hello from Lambda!',
            'action': action.item()  # integer action corresponding to discretized action
        })
    }
