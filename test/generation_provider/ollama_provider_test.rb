require "test_helper"
require "openai"
require "active_agent/action_prompt"
require "active_agent/generation_provider/ollama_provider"

class OllamaProviderTest < ActiveSupport::TestCase
  setup do
    @config = {
      "service" => "Ollama",
      "model" => "gemma3:latest",
      "host" => "http://localhost:11434",
      "api_version" => "v1",
      "embedding_model" => "nomic-embed-text"
    }
    @provider = ActiveAgent::GenerationProvider::OllamaProvider.new(@config)

    @prompt = ActiveAgent::ActionPrompt::Prompt.new(
      message: ActiveAgent::ActionPrompt::Message.new(content: "Test content for embedding"),
      instructions: "You are a test agent"
    )
  end

  test "initializes with correct configuration" do
    assert_equal "gemma3:latest", @provider.instance_variable_get(:@model_name)
    assert_equal "http://localhost:11434", @provider.instance_variable_get(:@host)
    assert_equal "v1", @provider.instance_variable_get(:@api_version)

    client = @provider.instance_variable_get(:@client)
    assert_instance_of OpenAI::Client, client
  end

  test "uses default values when config values not provided" do
    minimal_config = {
      "service" => "Ollama",
      "model" => "llama2:latest"
    }
    provider = ActiveAgent::GenerationProvider::OllamaProvider.new(minimal_config)

    assert_equal "http://localhost:11434", provider.instance_variable_get(:@host)
    assert_equal "v1", provider.instance_variable_get(:@api_version)
  end

  test "embeddings_parameters returns correct structure" do
    params = @provider.embeddings_parameters(input: "Test text", model: "nomic-embed-text")

    assert_equal "nomic-embed-text", params[:model]
    assert_equal "Test text", params[:prompt]
  end

  test "embeddings_parameters uses config embedding_model when available" do
    params = @provider.embeddings_parameters(input: "Test text")

    assert_equal "nomic-embed-text", params[:model]
    assert_equal "Test text", params[:prompt]
  end

  test "embeddings_parameters uses prompt message content by default" do
    @provider.instance_variable_set(:@prompt, @prompt)
    params = @provider.embeddings_parameters

    assert_equal "nomic-embed-text", params[:model]
    assert_equal "Test content for embedding", params[:prompt]
  end

  test "embeddings_response creates proper response object" do
    mock_response = {
      "embedding" => [ 0.1, 0.2, 0.3, 0.4, 0.5 ],
      "model" => "nomic-embed-text",
      "created" => 1234567890
    }

    request_params = {
      model: "nomic-embed-text",
      prompt: "Test text"
    }

    @provider.instance_variable_set(:@prompt, @prompt)
    response = @provider.embeddings_response(mock_response, request_params)

    assert_instance_of ActiveAgent::GenerationProvider::Response, response
    assert_equal @prompt, response.prompt
    assert_instance_of ActiveAgent::ActionPrompt::Message, response.message
    assert_equal [ 0.1, 0.2, 0.3, 0.4, 0.5 ], response.message.content
    assert_equal :assistant, response.message.role
    assert_equal mock_response, response.raw_response
    assert_equal request_params, response.raw_request
  end

  test "embed method works with Ollama provider" do
    VCR.use_cassette("ollama_provider_embed") do
      # Skip if Ollama is not running locally
      begin
        # region ollama_provider_embed
        provider = ActiveAgent::GenerationProvider::OllamaProvider.new(@config)
        prompt = ActiveAgent::ActionPrompt::Prompt.new(
          message: ActiveAgent::ActionPrompt::Message.new(content: "Generate an embedding for this text"),
          instructions: "You are an embedding test agent"
        )

        response = provider.embed(prompt)
        # endregion ollama_provider_embed

        assert_not_nil response
        assert_instance_of ActiveAgent::GenerationProvider::Response, response
        assert_not_nil response.message.content
        assert_kind_of Array, response.message.content
        assert response.message.content.all? { |val| val.is_a?(Numeric) }

        doc_example_output(response)
      rescue => e
        skip "Ollama is not running locally: #{e.message}"
      end
    end
  end

  test "inherits from OpenAIProvider" do
    assert ActiveAgent::GenerationProvider::OllamaProvider < ActiveAgent::GenerationProvider::OpenAIProvider
  end

  test "overrides embeddings methods from parent class" do
    # Verify that OllamaProvider has its own implementation of these methods
    assert @provider.respond_to?(:embeddings_parameters, true)
    assert @provider.respond_to?(:embeddings_response, true)

    # Verify the methods are defined in OllamaProvider, not just inherited
    ollama_methods = ActiveAgent::GenerationProvider::OllamaProvider.instance_methods(false)
    assert_includes ollama_methods, :embeddings_parameters
    assert_includes ollama_methods, :embeddings_response
  end

  test "handles Ollama-specific embedding format" do
    # Ollama returns embeddings in a different format than OpenAI
    # It returns a single "embedding" field instead of "data" array
    ollama_response = {
      "embedding" => [ 0.1, 0.2, 0.3 ],
      "model" => "nomic-embed-text"
    }

    @provider.instance_variable_set(:@prompt, @prompt)
    response = @provider.embeddings_response(ollama_response)

    assert_equal [ 0.1, 0.2, 0.3 ], response.message.content
  end
end
