export class ControllableEventSource {
  static instances: ControllableEventSource[] = [];

  onmessage: EventSource["onmessage"] = null;
  onerror: EventSource["onerror"] = null;
  closeCalls = 0;

  constructor(readonly url: string) {
    ControllableEventSource.instances.push(this);
  }

  close() {
    this.closeCalls += 1;
  }

  emitError(error: Event = new Event("error")) {
    this.onerror?.call(this as unknown as EventSource, error);
  }

  emitMessage(data: string) {
    return this.onmessage?.call(
      this as unknown as EventSource,
      { data } as MessageEvent,
    );
  }

  static reset() {
    ControllableEventSource.instances = [];
  }
}
