package none;

import java.io.IOException;
import java.net.URI;

import org.glassfish.grizzly.http.server.HttpServer;
import org.glassfish.jersey.grizzly2.httpserver.GrizzlyHttpServerFactory;
import org.glassfish.jersey.jackson.JacksonFeature;
import org.glassfish.jersey.server.ResourceConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);
    public static final String BASE_URI = "http://localhost:8080/";

    public static HttpServer startHttpServer() {
        final ResourceConfig resourceConfig = new ResourceConfig().packages("none.services")
                .register(JacksonFeature.class)
                .property("jersey.config.server.wadl.disableWadl", true);
        return GrizzlyHttpServerFactory.createHttpServer(URI.create(BASE_URI), resourceConfig);
    }

    public static void main(String[] args) throws IOException {
        final HttpServer server = startHttpServer();
        logger.info("Jersey app started with WADL available at {}application.wadl\nHit enter to stop it...", BASE_URI);
        System.in.read();
        server.shutdownNow();
        logger.info("Server stopped.");
    }
}
