#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "getpid.h"
#include "ccg-helper.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace dynet;

struct ModelVars{
 public:
  unsigned LAYERS;
  unsigned CHAR_INPUT_DIM;
  unsigned WORD_INPUT_DIM;
  unsigned LSTM_CHAR_OUTPUT_DIM;
  unsigned HIDDEN_DIM;
  unsigned ACTION_DIM;
  unsigned PRETRAINED_DIM;
  unsigned LSTM_INPUT_DIM;
  unsigned POS_DIM;
  unsigned REL_DIM;

  bool use_dropout;
  double dropout_rate;

  string pretrained_embedding_filepath;
  string model_filepath;

  string to_string(){
    ostringstream os;
    os << "Model Vars:" << "\n"
       << "use dropout: " << use_dropout << "\n"
       << "dropout_rate: " << dropout_rate << "\n"
       << "CHAR_INPUT_DIM: " << CHAR_INPUT_DIM << "\n"
       << "WORD_INPUT_DIM: " << WORD_INPUT_DIM << "\n"
       << "LSTM_CHAR_OUTPUT_DIM: " << LSTM_CHAR_OUTPUT_DIM << "\n"
       << "HIDDEN_DIM: " << HIDDEN_DIM << "\n"
       << "ACTION_DIM: " << ACTION_DIM << "\n"
       << "PRETRAINED_DIM: " << PRETRAINED_DIM << "\n"
       << "LSTM_INPUT_DIM: " << LSTM_INPUT_DIM << "\n"
       << "POS_DIM: " << POS_DIM << "\n"
       << "REL_DIM: " << REL_DIM << "\n"
       << "pretrained_embedding_filepath: " << pretrained_embedding_filepath << "\n"
       << "model_filepath: " << model_filepath << "\n"; 
    return os.str();
  }

  void counter_clean(){
    correct = 0.0;
    total = 0.0;
  }
  void counter_correct_plus_one() {
    correct = correct + 1;
  }
  void counter_total_plus_one() {
    total = total + 1;
  }
  float current_acc() {
    return (correct/total);
  }
 private:
  float correct = 0.0;
  float total = 0.0;
};

struct ParserBuilder {
  explicit ParserBuilder(Model* model, ModelVars* mv, DictMap* dict_map) :
    // TODO: CHECK IF THIS WAY WORKS
    mv_(mv),
    dict_map_(dict_map),
    // Parameter for the LSTMs
    fw_lstm(mv_->LAYERS, mv_->LSTM_INPUT_DIM, mv->LSTM_INPUT_DIM, model),
    bw_lstm(mv_->LAYERS, mv_->LSTM_INPUT_DIM, mv->LSTM_INPUT_DIM, model),
    // Parameter for the char-LSTMs
    p_start_of_word(model->add_parameters({mv_->CHAR_INPUT_DIM})),
    p_end_of_word(model->add_parameters({mv_->CHAR_INPUT_DIM})), 
    fw_char_lstm(mv_->LAYERS, mv_->CHAR_INPUT_DIM, mv_->LSTM_CHAR_OUTPUT_DIM, model),
    bw_char_lstm(mv_->LAYERS, mv_->CHAR_INPUT_DIM, mv_->LSTM_CHAR_OUTPUT_DIM, model),
    // Parameter for word, pretrained, and char
    p_w(model->add_lookup_parameters(dict_map_->vocab_size(), {mv_->WORD_INPUT_DIM})),
    p_c(model->add_lookup_parameters(dict_map_->char_size(), {mv_->CHAR_INPUT_DIM})),
    p_t(model->add_lookup_parameters(dict_map_->lc_vocab_size(), {mv_->PRETRAINED_DIM})),
    // Parameter for the concatenation of input, char_fwd, char_bwd, word, pos, pretrain
    p_x2i(model->add_parameters({mv_->LSTM_INPUT_DIM, (2 * mv_->LSTM_CHAR_OUTPUT_DIM + mv_->WORD_INPUT_DIM + mv_->PRETRAINED_DIM)})),
    p_x2ib(model->add_parameters({mv_->LSTM_INPUT_DIM})),
    p_th2t(model->add_parameters({dict_map_->supertag_size(), mv->LSTM_INPUT_DIM * 2})),
    p_tbias(model->add_parameters({dict_map_->supertag_size()}))
    {
      // Set up pretrained embedding
      // NOTE: THE pretrained embedding is lower cased here!
      load_embedding(dict_map_->get_dict(DictMap::DICTMAP_IND_LC_TERM), mv_->pretrained_embedding_filepath, p_t);
    }

  Expression log_prob_parser(ComputationGraph& cg, Sentence& sent, bool build_training_graph, vector<unsigned int> &results) {
    // Set up the two LSTMs
    fw_lstm.new_graph(cg);
    if(mv_->use_dropout){
      fw_lstm.set_dropout(mv_->dropout_rate);
    }else{
      fw_lstm.disable_dropout();
    }
    fw_lstm.start_new_sequence();
    bw_lstm.new_graph(cg);
    if(mv_->use_dropout){
      bw_lstm.set_dropout(mv_->dropout_rate);
    }else{
      bw_lstm.disable_dropout();
    }
    bw_lstm.start_new_sequence();

    Expression x2ib = parameter(cg, p_x2ib);
    Expression x2i = parameter(cg, p_x2i);
    Expression word_start = parameter(cg, p_start_of_word);
    Expression word_end = parameter(cg, p_end_of_word);

    // set up char LSTMs
    fw_char_lstm.new_graph(cg);
    bw_char_lstm.new_graph(cg);

    // prepare the input tokens
    vector<Expression> x;
    for (int i = 0; i < sent.size(); ++i){
      // the input is constructed by three parts, word form, charcter embedding, and pretrain word embedding
      
      vector<Expression> input_comps;

      Expression w_c_f; // the char level model of the word (forward)
      Expression w_c_b; // the char level model of the word (backward)
      Expression w; // the look up of the word

      fw_char_lstm.start_new_sequence();
      fw_char_lstm.add_input(word_start);
      for (int char_ind = 0; char_ind < static_cast<int>(sent.chars[i].size()); ++char_ind){
        Expression char_e = lookup(cg, p_c, sent.chars[i][char_ind]);
        fw_char_lstm.add_input(char_e);
      }
      fw_char_lstm.add_input(word_end);

      // Add the backward embedding
      bw_char_lstm.start_new_sequence();
      bw_char_lstm.add_input(word_end);
      for (int char_ind = sent.chars[i].size() - 1; char_ind >= 0; --char_ind){
        Expression char_e = lookup(cg, p_c, sent.chars[i][char_ind]);
        bw_char_lstm.add_input(char_e);
      }
      bw_char_lstm.add_input(word_start);
      w_c_f = fw_char_lstm.back();
      w_c_b = bw_char_lstm.back();

      w = lookup(cg, p_w, sent.terms[i]);

      input_comps.push_back(w_c_f);
      input_comps.push_back(w_c_b);
      input_comps.push_back(w);

      // Add the pretrained embedding to the input representation
      Expression pertrain_emb = const_lookup(cg, p_t, sent.lower_cased_terms[i]);
      input_comps.push_back(pertrain_emb);

      Expression input_comps_exp = concatenate(input_comps);
      Expression input_word = rectify(affine_transform({x2ib, x2i, input_comps_exp}));
      x.push_back(input_word);
    }

    const int len = x.size();
    vector<Expression> fwd(len), rev(len), res(len);
    for (int i = 0; i < len; ++i){
      fwd[i] = fw_lstm.add_input(x[i]);
    }
    for (int i = len - 1; i >= 0; --i){
      rev[i] = bw_lstm.add_input(x[i]);
    }
    for (int i = 0; i < len; ++i){
      vector<Expression> cat = {fwd[i], rev[i]};
      res[i] = concatenate(cat);
    }

    Expression i_tbias = parameter(cg, p_tbias);
    Expression i_th2t = parameter(cg, p_th2t);

    // compute the loss
    vector<Expression> errs;
    for (int i = 0; i < len; ++i){
      Expression i_t = affine_transform({i_tbias, i_th2t, res[i]});

      // get the prediction at this step
      {
        vector<float> dist = as_vector(cg.incremental_forward(i_t));
        double best = -9e99;
        int besti = -1;
        for (int i = 0; i < dist.size(); ++i) {
          if (dist[i] > best) { best = dist[i]; besti = i; }
        }
        assert(besti >= 0);
        results.push_back(besti);

        if(!dict_map_->is_dummy_supertag(sent.supertags[i])){
          if (sent.supertags[i] == besti) {
            mv_->counter_correct_plus_one();
          }
          mv_->counter_total_plus_one();
          Expression i_err = pickneglogsoftmax(i_t, sent.supertags[i]);
          errs.push_back(i_err);
        }
      }

    }
    return sum(errs);


  }

  DictMap* get_dict_map(){
    return dict_map_;
  }

 private:
  ModelVars* mv_;
  DictMap* dict_map_;

  // parameter
  LSTMBuilder fw_lstm;
  LSTMBuilder bw_lstm;

  Parameter p_start_of_word;
  Parameter p_end_of_word;

  LSTMBuilder fw_char_lstm;
  LSTMBuilder bw_char_lstm;

  LookupParameter p_w; // word embeddings
  Parameter p_root_in_char;
  LookupParameter p_c; // lookup for chars
  LookupParameter p_t; // pretrained word embeddings (not updated)
  Parameter p_x2i; // char, pos, word, pretrain to input
  Parameter p_x2ib; // bias for char, pos, word, pretrain to input

  Parameter p_th2t;
  Parameter p_tbias;

};





namespace po = boost::program_options;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data", po::value<string>(), "Development corpus")
        ("test_data", po::value<string>(), "Test corpus")
        ("model", po::value<string>()->default_value(""), "Load saved model from this file")
        // ("adamhyper", po::value<double>()->default_value(5e-4), "Step size for adam trainer")
        ("step_size", po::value<double>()->default_value(0.1), "Step size for simple sgd trainer")
        ("decay_rate", po::value<double>()->default_value(0.05), "Decay rate for simple sgd trainer")
        ("report_every_i", po::value<unsigned>()->default_value(100), "input embedding size")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("word_input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("char_input_dim", po::value<unsigned>()->default_value(10), "char input size")
        ("lstm_char_output_dim", po::value<unsigned>()->default_value(32), "the embedding size contributed from char-lstm")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("action_dim", po::value<unsigned>()->default_value(32), "action embedding size")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(128), "the input dimension to LSTMs (stack, buffer, action)")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
        ("rel_dim", po::value<unsigned>()->default_value(32), "relation dimension")
        ("pretrained_embedding_filepath", po::value<string>()->default_value(""), "Pretrained word embeddings")
        ("predefined_tags", po::value<string>()->default_value(""), "the predefined 425 supertags")
        ("dropout_rate", po::value<double>()->default_value(0), "dropout_rate")
        ("help,h", "Help");

  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
}

void predict(ParserBuilder& parser_builder, vector<Sentence> corpus, string output){
  ofstream o(output);
  for (int i = 0; i < static_cast<int>(corpus.size()); ++i){
    ComputationGraph cg;
    auto& sent = corpus[i];
    vector<unsigned int> results;
    parser_builder.log_prob_parser(cg, sent, false, results);
    for (int j = 0; j < static_cast<int>(results.size()); ++j){
      string supertag = ((parser_builder.get_dict_map())->get_dict(DictMap::DICTMAP_IND_SUPERTAG)).convert(results[j]);
      o << supertag;
      if (j == static_cast<int>(results.size() - 1)){
        o << "\n";
      } else {
        o << " ";
      }
    }
  }
  o.close();
}

float eval_parses(string gold_file_path, string sys_file_path){
  string eval_file_name = std::tmpnam(nullptr);

  // Call Eval script
  cerr << "running evaluation step." << endl;
  string eval_command = "python ../../scripts/eval_parses.py " + gold_file_path + " " + sys_file_path + " > " + eval_file_name;
  const char* eval_cmd = eval_command.c_str();
  system(eval_cmd);

  std::ifstream infile(eval_file_name);
  assert(infile);
  std::string line;

  float score = 0;

  while (std::getline(infile, line)) {
    boost::algorithm::trim(line);
    if (line.length() == 0) continue;
    score = std::stof(line);
    break;
  }

  string rm_temp_file = "rm -f " + eval_file_name;
  const char* rm_cmd = rm_temp_file.c_str();
  system(rm_cmd);

  return score;
}

void save_current(string sys_file_path){
  string cp_file = "cp " + sys_file_path + " " + sys_file_path + "_current_best";
  const char* cp_cmd = cp_file.c_str();
  system(cp_cmd);
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);

  // Initialize the context
  DictMap dict_map;
  ModelVars mv;

  mv.LAYERS = conf["layers"].as<unsigned>();
  mv.WORD_INPUT_DIM = conf["word_input_dim"].as<unsigned>();
  mv.CHAR_INPUT_DIM = conf["char_input_dim"].as<unsigned>();
  mv.LSTM_CHAR_OUTPUT_DIM = conf["lstm_char_output_dim"].as<unsigned>();
  mv.HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  mv.ACTION_DIM = conf["action_dim"].as<unsigned>();
  mv.PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  mv.LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  mv.POS_DIM = conf["pos_dim"].as<unsigned>();
  mv.REL_DIM = conf["rel_dim"].as<unsigned>();
  mv.pretrained_embedding_filepath = conf["pretrained_embedding_filepath"].as<string>();
  mv.model_filepath = conf["model"].as<string>();
  mv.dropout_rate = conf["dropout_rate"].as<double>();
  if (mv.dropout_rate > 0) {
    mv.use_dropout = true;
  } else{
    mv.use_dropout = false;
  }

  cout << mv.to_string() << endl;

  // read the predefined tags into the dict_map
  dict_map.set_predefined_supertags(read_predefined_supertags(conf["predefined_tags"].as<string>()));

  // start read the corpus
  vector<Sentence> corpus_train = read_from_file(conf["training_data"].as<string>(), dict_map);
  cerr << "Reading " << corpus_train.size() << " sentences from " << conf["training_data"].as<string>() << endl;

  vector<Sentence> corpus_dev = read_from_file(conf["dev_data"].as<string>(), dict_map);
  cerr << "Reading " << corpus_dev.size() << " sentences from " << conf["dev_data"].as<string>() << endl;

  vector<Sentence> corpus_test = read_from_file(conf["test_data"].as<string>(), dict_map);
  cerr << "Reading " << corpus_test.size() << " sentences from " << conf["test_data"].as<string>() << endl;

   // Log the dict map status
  cerr << dict_map.to_string() << endl;

  Model model;
  model.set_weight_decay_lambda(1e-6);
  ParserBuilder parser_builder(&model, &mv, &dict_map);

  // auto sgd = new AdamTrainer(&model, conf["adamhyper"].as<double>(), 0.01, 0.9999, 1e-8);
  auto sgd = new SimpleSGDTrainer(&model, conf["step_size"].as<double>(), conf["decay_rate"].as<double>());
  cerr << "Using SimpleSGDTrainer with step size " << conf["step_size"].as<double>() << " and decay rate " << conf["decay_rate"].as<double>() << endl;
  
  unsigned si = corpus_train.size();
  vector<unsigned> order(corpus_train.size());
  for (int i = 0; i < static_cast<int>(order.size()); ++i) {
    order[i] = i;
  }
  bool first = true;
  int report = 0;

  float current_best_dev = 0.0;
  float current_best_test = 0.0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0.0;
    for (int i = 0; i < 100; ++i) {
      if (si == corpus_train.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        // cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }
      // construct graph
      ComputationGraph cg;
      auto& sent = corpus_train[order[si]];
      ++si;
      vector<unsigned int> results;
      Expression loss_exp =  parser_builder.log_prob_parser(cg, sent, true, results);
      loss += as_scalar(cg.forward(loss_exp));
      cg.backward(loss_exp);
      sgd->update(1.0);
    }
    sgd->status();
    cerr << "acc = " << mv.current_acc() << endl;
    mv.counter_clean();
    report++;
    if (report % (conf["report_every_i"].as<unsigned>()) == 0) {
       // Begin the training loop
      string dev_output = mv.model_filepath + ".dev.pred";
      string test_output = mv.model_filepath + ".test.pred";
      // For simplicity, we just reuse the parser builder here, but it should be totally okay when you
      // save mv, model and dictmap and rebuild the parser builder
      predict(parser_builder, corpus_dev, dev_output);
      float dev_score = eval_parses(conf["dev_data"].as<string>(), dev_output);
      cerr << "score dev: " << dev_score << endl;
      predict(parser_builder, corpus_test, test_output);
      float test_score = eval_parses(conf["test_data"].as<string>(), test_output);
      cerr << "score test: " << eval_parses(conf["test_data"].as<string>(), test_output) << endl;
      // only compare dev score
      if (dev_score > current_best_dev){
        current_best_dev = dev_score;
        current_best_test = test_score;
        save_current(dev_output);
        save_current(test_output);
      }
      cerr << "current best dev: " << current_best_dev << " current best test: " << current_best_test << endl;
    }
  }
}
