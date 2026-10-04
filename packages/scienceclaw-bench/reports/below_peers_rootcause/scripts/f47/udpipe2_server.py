#!/usr/bin/env python3

# This file is part of UDPipe 2.0 <http://github.com/ufal/udpipe>.
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse
import contextlib
import email.parser
import http.server
import itertools
import json
import os
import socketserver
import sys
import threading
import time
import unicodedata
import urllib.parse

import udpipe2
import udpipe2_dataset
import ufal.udpipe
import wembedding_service.wembeddings.wembeddings as wembeddings

__version__ = "2.1.1-dev"


class TooLongError(Exception):
    pass

class Models:
    class Model:
        class Network:
            _mutex = threading.Lock()

            def __init__(self, path, server_args):
                self._path = path
                self._server_args = server_args
                self.network, self.args, self.train = None, None, None

            def load(self):
                if self.network is not None:
                    return
                with self._mutex:
                    if self.network is not None:
                        return

                    with open(os.path.join(self._path, "options.json"), mode="r") as options_file:
                        self.args = argparse.Namespace(**json.load(options_file))
                    udpipe2.UDPipe2.postprocess_arguments(self.args)
                    self.args.batch_size = self._server_args.batch_size

                    self.train = udpipe2_dataset.UDPipe2Dataset.load_mappings(os.path.join(self._path, "mappings.pickle"))
                    self.network = udpipe2.UDPipe2(threads=self._server_args.threads)
                    self.network.construct(self.args, self.train, [], [], predict_only=True)
                    self.network.load(self._path, self.args.morphodita)

                    print("Loaded model {}".format(os.path.basename(self._path)), file=sys.stderr, flush=True)


        def __init__(self, names, path, network, variant, acknowledgements, server_args):
            self.names = names
            self.acknowledgements = acknowledgements
            self._network = network
            self._variant = variant
            self._server_args = server_args

            # Load the tokenizer
            tokenizer_path = os.path.join(path, "{}.tokenizer".format(variant))
            self._tokenizer = ufal.udpipe.Model.load(tokenizer_path)
            if self._tokenizer is None:
                raise RuntimeError("Cannot load tokenizer from {}".format(tokenizer_path))

            self._conllu_input = ufal.udpipe.InputFormat.newConlluInputFormat()
            if self._conllu_input is None:
                raise RuntimeError("Cannot create CoNLL-U input format")

            self._conllu_output = ufal.udpipe.OutputFormat.newConlluOutputFormat()
            if self._conllu_output is None:
                raise RuntimeError("Cannot create CoNLL-U output format")

            # Load the network if requested
            if names[0] in server_args.preload_models or "all" in server_args.preload_models:
                self._network.load()

        def read(self, text, input_format):
            reader = ufal.udpipe.InputFormat.newInputFormat(input_format)
            if reader is None:
                raise RuntimeError("Unknown input format '{}'".format(input_format))
            # Do not return a generator, but a list to raise exceptions early
            return list(self._read(text, reader))

        def tokenize(self, text, tokenizer_options):
            tokenizer = self._tokenizer.newTokenizer(tokenizer_options)
            if tokenizer is None:
                raise RuntimeError("Cannot create tokenizer.")
            # Do not return a generator, but a list to raise exceptions early
            return list(self._read(text, tokenizer))

        def _read(self, text, reader):
            sentence = ufal.udpipe.Sentence()
            processing_error = ufal.udpipe.ProcessingError()
            reader.setText(text)
            while reader.nextSentence(sentence, processing_error):
                if len(sentence.words) > 1001:
                    raise TooLongError()
                yield sentence
                sentence = ufal.udpipe.Sentence()
            if processing_error.occurred():
                raise RuntimeError("Cannot read input data: '{}'".format(processing_error.message))

        def create_writer(self, output_format):
            writer = ufal.udpipe.OutputFormat.newOutputFormat(output_format)
            if writer is None:
                raise RuntimeError("Unknown output format '{}'".format(output_format))
            return writer

        def predict(self, sentences, tag, parse, writer):
            # Run the model
            if tag or parse:
                # Load the network if it has not been loaded already
                self._network.load()

                wembedding_input, conllu_input = [], []
                for sentence in sentences:
                    wembedding_input.append([word.form for word in sentence.words[1:]])
                    conllu_input.append(self._conllu_output.writeSentence(sentence))

                time_wes = time.time()
                # Compute the WEmbeddings
                with self._server_args.optional_semaphore:
                    time_we = time.time()
                    if self._network.args.wembedding_model:
                        wembeddings = self._server_args.wembedding_server.compute_embeddings(self._network.args.wembedding_model, wembedding_input)
                    else:
                        wembeddings = []

                time_ds = time.time()
                # Create UDPipe2Dataset
                dataset = udpipe2_dataset.UDPipe2Dataset(text="".join(conllu_input), train=self._network.train, shuffle_batches=False,
                                                         embeddings=wembeddings, override_variant=self._variant)

                # Prepare network arguments
                network_args = argparse.Namespace(**vars(self._network.args))
                if not tag: network_args.tags = []
                if not parse: network_args.parse = 0

                # Perform the prediction
                time_nws = time.time()
                with self._server_args.optional_semaphore:
                    time_nw = time.time()
                    predicted = self._network.network.predict(dataset, evaluating=False, args=network_args)
                    time_rd = time.time()

                # Load the predicted CoNLL-U to ufal.udpipe sentences
                sentences = self._read(predicted, self._conllu_input)

                print("Request, WE {:.2f}+{:.2f}ms,".format(1000 * (time_ds - time_we), 1000 * (time_we - time_wes)),
                      "DS {:.2f}ms,".format(1000 * (time_nws - time_ds)),
                      "NW {:.2f}+{:.2f}ms,".format(1000 * (time_rd - time_nw), 1000 * (time_nw - time_nws)),
                      "RD {:.2f}ms.".format(1000 * (time.time() - time_rd)),
                      file=sys.stderr, flush=True)

            # Generate output
            output = []
            for sentence in sentences:
                output.append(writer.writeSentence(sentence))
            output.append(writer.finishDocument())
            return "".join(output)

    def __init__(self, server_args):
        self.default_model = server_args.default_model
        self.models_list = []
        self.models_by_names = {}
        networks_by_path = {}

        for i in range(0, len(server_args.models), 4):
            names, path, variant, acknowledgements = server_args.models[i:i+4]
            names = names.split(":")
            names = [name.split("-") for name in names]
            names = ["-".join(parts[:None if not i else -i]) for parts in names for i in range(len(parts))]

            if not path in networks_by_path:
                networks_by_path[path] = self.Model.Network(path, server_args)
            self.models_list.append(self.Model(names, path, networks_by_path[path], variant, acknowledgements, server_args))
            for name in names:
                self.models_by_names.setdefault(name, self.models_list[-1])

        # Check the default model exists
        assert self.default_model in self.models_by_names


class UDServer(socketserver.ThreadingTCPServer):
    class UDServerRequestHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def respond(request, content_type, code=200, additional_headers={}):
            request.close_connection = True
            request.send_response(code)
            request.send_header("Connection", "close")
            request.send_header("Content-Type", content_type)
            request.send_header("Access-Control-Allow-Origin", "*")
            for key, value in additional_headers.items():
                request.send_header(key, value)
            request.end_headers()

        def respond_error(request, message, code=400):
            request.respond("text/plain", code)
            request.wfile.write(message.encode("utf-8"))

        def do_GET(request):
            # Parse the URL
            params = {}
            try:
                request.path = request.path.encode("iso-8859-1").decode("utf-8")
                url = urllib.parse.urlparse(request.path)
                for name, value in urllib.parse.parse_qsl(url.query, encoding="utf-8", keep_blank_values=True, errors="strict"):
                    params[name] = value
            except:
                return request.respond_error("Cannot parse request URL.")

            # Parse the body of a POST request
            if request.command == "POST":
                if request.headers.get("Transfer-Encoding", "identity").lower() != "identity":
                    return request.respond_error("Only 'identity' Transfer-Encoding of payload is supported for now.")

                try:
                    content_length = int(request.headers["Content-Length"])
                except:
                    return request.respond_error("The Content-Length of payload is required.")

                if content_length > request.server._server_args.max_request_size:
                    return request.respond_error("The payload size is too large.")

                # Raw text on input for weblicht
                if url.path.startswith("/weblicht/"):
                    # Ignore all but `model` GET param
                    params = {"model": params["model"]} if "model" in params else {}

                    try:
                        params["data"] = request.rfile.read(content_length).decode("utf-8")
                    except:
                        return request.respond_error("The payload is not in UTF-8 encoding.")

                    if url.path == "/weblicht/tokenize": params["tokenizer"] = ""
                    else: params["input"] = "conllu"
                    params["output"] = "conllu"
                    if url.path == "/weblicht/tag": params["tagger"] = ""
                    if url.path == "/weblicht/parse": params["parser"] = ""
                # multipart/form-data
                elif request.headers.get("Content-Type", "").startswith("multipart/form-data"):
                    try:
                        parser = email.parser.BytesFeedParser()
                        parser.feed(b"Content-Type: " + request.headers["Content-Type"].encode("ascii") + b"\r\n\r\n")
                        while content_length:
                            parser.feed(request.rfile.read(min(content_length, 4096)))
                            content_length -= min(content_length, 4096)
                        for part in parser.close().get_payload():
                            name = part.get_param("name", header="Content-Disposition")
                            if name:
                                params[name] = part.get_payload(decode=True).decode("utf-8")
                    except:
                        return request.respond_error("Cannot parse the multipart/form-data payload.")
                # application/x-www-form-urlencoded
                elif request.headers.get("Content-Type", "").startswith("application/x-www-form-urlencoded"):
                    try:
                        for name, value in urllib.parse.parse_qsl(
                                request.rfile.read(content_length).decode("utf-8"), encoding="utf-8", keep_blank_values=True, errors="strict"):
                            params[name] = value
                    except:
                        return request.respond_error("Cannot parse the application/x-www-form-urlencoded payload.")
                else:
                    return request.respond_error("Unsupported payload Content-Type '{}'.".format(request.headers.get("Content-Type", "<none>")))

            # Handle /models
            if url.path == "/models":
                response = {
                    "models": {model.names[0]: ["tokenizer", "tagger", "parser"] for model in request.server._models.models_list},
                    "default_model": request.server._models.default_model,
                }
                request.respond("application/json")
                request.wfile.write(json.dumps(response, indent=1).encode("utf-8"))
            # Handle /process
            elif url.path in ["/process", "/weblicht/tokenize", "/weblicht/tag", "/weblicht/parse"]:
                weblicht = url.path.startswith("/weblicht")

                if "data" not in params:
                    return request.respond_error("The parameter 'data' is required.")
                params["data"] = unicodedata.normalize("NFC", params["data"])

                model = params.get("model", request.server._models.default_model)
                if model not in request.server._models.models_by_names:
                    return request.respond_error("The requested model '{}' does not exist.".format(model))
                model = request.server._models.models_by_names[model]

                # Start by reading and optionally tokenizing the input data.
                if "tokenizer" in params:
                    try:
                        sentences = model.tokenize(params["data"], params["tokenizer"])
                    except TooLongError:
                        return request.respond_error("During tokenization, sentence longer than 1000 words was found, aborting.\nThat should only happen with presegmented input.\nPlease make sure you do not generate such long sentences.\n")
                    except:
                        return request.respond_error("An error occured during tokenization of the input.")
                else:
                    try:
                        sentences = model.read(params["data"], params.get("input", "conllu"))
                    except TooLongError:
                        return request.respond_error("Sentence longer than 1000 words was found on input, aborting.\nPlease make sure the input sentences have at most 1000 words.\n")
                    except:
                        return request.respond_error("Cannot parse the input in '{}' format.".format(params.get("input", "conllu")))
                infclen = sum(sum(len(word.form) for word in sentence.words[1:]) for sentence in sentences)

                # Create the writer
                output_format = params.get("output", "conllu")
                try:
                    writer = model.create_writer(output_format)
                except:
                    return request.respond_error("Unknown output format '{}'.".format(output_format))

                # Process the data
                tag, parse, output_format = "tagger" in params, "parser" in params, params.get("output", "conllu")
                batch, started_responding = [], False
                try:
                    for sentence in itertools.chain(sentences, ["EOF"]):
                        if sentence == "EOF" or len(batch) == request.server._server_args.batch_size:
                            output = model.predict(batch, tag, parse, writer)
                            if not started_responding:
                                # The first batch is ready, we commit to generate output.
                                started_responding=True
                                if weblicht:
                                    request.respond("application/conllu")
                                else:
                                    request.respond("application/json", additional_headers={"X-Billing-Input-NFC-Len": str(infclen)})
                                    request.wfile.write(json.dumps({
                                        "model": model.names[0],
                                        "acknowledgements": ["http://ufal.mff.cuni.cz/udpipe/2#udpipe2_acknowledgements", model.acknowledgements],
                                        "result": "",
                                    }, indent=1)[:-3].encode("utf-8"))
                                    if output_format == "conllu":
                                        request.wfile.write(json.dumps(
                                            "# generator = UDPipe 2, https://lindat.mff.cuni.cz/services/udpipe\n"
                                            "# udpipe_model = {}\n"
                                            "# udpipe_model_licence = CC BY-NC-SA\n".format(model.names[0]))[1:-1].encode("utf-8"))
                            if weblicht:
                                request.wfile.write(output.encode("utf-8"))
                            else:
                                request.wfile.write(json.dumps(output, ensure_ascii=False)[1:-1].encode("utf-8"))
                            batch = []
                        batch.append(sentence)
                    if not weblicht:
                        request.wfile.write(b'"\n}\n')
                except:
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()

                    if not started_responding:
                        request.respond_error("An internal error occurred during processing.")
                    else:
                        if weblicht:
                            request.wfile.write(b'\n\nAn internal error occurred during processing, producing incorrect CoNLL-U!')
                        else:
                            request.wfile.write(b'",\n"An internal error occurred during processing, producing incorrect JSON!"')
            # Unknown URL
            else:
                request.respond_error("No handler for the given URL '{}'".format(url.path), code=404)

        def do_POST(request):
            return request.do_GET()

    daemon_threads = False

    def __init__(self, server_args, models):
        super().__init__(("", server_args.port), self.UDServerRequestHandler)

        self._server_args = server_args
        self._models = models

    def server_bind(self):
        import socket
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()

    def service_actions(self):
        if isinstance(getattr(self, "_threads", None), list):
            if len(self._threads) >= 1024:
                self._threads = [thread for thread in self._threads if thread.is_alive()]


if __name__ == "__main__":
    import signal
    import threading

    # Parse server arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to use")
    parser.add_argument("default_model", type=str, help="Default model")
    parser.add_argument("models", type=str, nargs="+", help="Models to serve")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--concurrent", default=None, type=int, help="Concurrent computations of NN")
    parser.add_argument("--logfile", default=None, type=str, help="Log path")
    parser.add_argument("--max_request_size", default=4096*1024, type=int, help="Maximum request size")
    parser.add_argument("--preload_models", default=[], nargs="*", type=str, help="Models to preload, or `all`")
    parser.add_argument("--threads", default=0, type=int, help="Threads to use")
    parser.add_argument("--wembedding_preload_models", default=[], nargs="*", type=str, help="WEmbedding models to preload")
    parser.add_argument("--wembedding_server", default=None, type=str, help="Address of an WEmbedding server")
    args = parser.parse_args()

    # Log stderr to logfile if given
    if args.logfile is not None:
        sys.stderr = open(args.logfile, "a", encoding="utf-8")

    # Create the WEmbeddings client
    if args.wembedding_server is not None:
        args.wembedding_server = wembeddings.WEmbeddings.ClientNetwork(args.wembedding_server)
    else:
        args.wembedding_server = wembeddings.WEmbeddings(threads=args.threads, preload_models=args.wembedding_preload_models)

    # Load the models
    models = Models(args)

    # Create a semaphore if needed
    args.optional_semaphore = threading.Semaphore(args.concurrent) if args.concurrent is not None else contextlib.nullcontext()

    # Create the server
    server = UDServer(args, models)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("Started UDPipe 2 server on port {}.".format(args.port), file=sys.stderr)
    print("To stop it gracefully, either send SIGINT (Ctrl+C) or SIGUSR1.", file=sys.stderr, flush=True)

    # Wait until the server should be closed
    signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT, signal.SIGUSR1])
    signal.sigwait([signal.SIGINT, signal.SIGUSR1])
    print("Initiating shutdown of the UDPipe 2 server.", file=sys.stderr, flush=True)
    server.shutdown()
    print("Stopped handling new requests, processing all current ones.", file=sys.stderr, flush=True)
    server.server_close()
    print("Finished shutdown of the UDPipe 2 server.", file=sys.stderr, flush=True)
