require "openai"
require_relative "open_ai_provider"

module ActiveAgent
  module GenerationProvider
    class OllamaProvider < OpenAIProvider
      def initialize(config)
        @config = config
        @access_token ||= config["api_key"] || config["access_token"] || ENV["OLLAMA_API_KEY"] || ENV["OLLAMA_ACCESS_TOKEN"]
        @model_name = config["model"]
        @host = config["host"] || "http://localhost:11434"
        @api_version = config["api_version"] || "v1"
        @client = OpenAI::Client.new(uri_base: @host, access_token: @access_token, log_errors: true, api_version: @api_version)
      end

      def embeddings_parameters(input: prompt.message.content, model: "text-embedding-3-large")
        {
          model: self.config["embedding_model"] || model,
          prompt: input
        }
      end


    end
  end
end
